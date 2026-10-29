"""
test_cloud.py — R2 job layout, temporary credentials, RunPod pod lifecycle (CPU pods by default) and the
pod entry point of the face pipeline, with an in-memory S3 and a fake RunPod API (no network).
"""

import io
from datetime import datetime, timedelta, timezone
import json
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
from botocore.exceptions import ClientError

from cloud import cloudflare_r2 as r2
from cloud import runpod as rp


class FakeS3:
    """The subset of the boto3 S3 client the module uses, backed by a dict."""

    def __init__(self):
        self.objects = {}
        self.mtimes = {}          # (bucket, key) -> LastModified; objects without one count as written "now"

    def upload_file(self, file_name, bucket, key):
        self.objects[(bucket, key)] = Path(file_name).read_bytes()

    def download_file(self, bucket, key, file_name):
        if (bucket, key) not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "GetObject")
        Path(file_name).write_bytes(self.objects[(bucket, key)])

    def put_object(self, Bucket, Key, Body, **_):
        self.objects[(Bucket, Key)] = Body if isinstance(Body, bytes) else Body.encode()

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {}

    def delete_object(self, Bucket, Key):
        self.objects.pop((Bucket, Key), None)

    def delete_objects(self, Bucket, Delete):
        for o in Delete["Objects"]:
            self.objects.pop((Bucket, o["Key"]), None)
        return {}

    def get_paginator(self, _name):
        store = self.objects

        mtimes = self.mtimes

        class Pager:
            def paginate(self, Bucket, Prefix=""):
                now = datetime.now(timezone.utc)
                yield {"Contents": [{"Key": k, "LastModified": mtimes.get((b, k), now)}
                                    for (b, k) in sorted(store) if b == Bucket and k.startswith(Prefix)]}
        return Pager()


class FakeRunPod:
    """create / get / delete pod; get_pod reports RUNNING until `exit_after` polls."""

    def __init__(self, fail_creates=0, exit_after=None, delete_fails=False):
        self.created, self.deleted, self.bodies = [], [], []
        self.fail_creates = fail_creates
        self.exit_after = exit_after
        self.delete_fails = delete_fails
        self.polls = 0

    def create_pod(self, body):
        self.bodies.append(body)
        if self.fail_creates > 0:
            self.fail_creates -= 1
            raise rp.RunPodError("no capacity")
        pod_id = f"pod{len(self.created) + 1}"
        self.created.append(pod_id)
        if body["computeType"] == "CPU":
            return {"id": pod_id, "memoryInGb": 2 * body["vcpuCount"], "vcpuCount": body["vcpuCount"], "costPerHr": 0.2}
        return {"id": pod_id, "gpu": {"displayName": "RTX A4000"}, "memoryInGb": body["minRAMPerGPU"],
                "vcpuCount": body["minVCPUPerGPU"], "costPerHr": 0.36}

    def get_pod(self, pod_id):
        self.polls += 1
        state = "EXITED" if self.exit_after is not None and self.polls >= self.exit_after else "RUNNING"
        return {"id": pod_id, "desiredStatus": state}

    def delete_pod(self, pod_id):
        if self.delete_fails:
            raise rp.RunPodError("network")
        self.deleted.append(pod_id)


BUCKET = "test-bucket"


class TestR2Layout(unittest.TestCase):
    def setUp(self):
        self.s3 = FakeS3()
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_delete_stale_jobs_only_touches_old_job_folders(self):
        """Old failed jobs go; a job written recently (may be running), the current job and other prefixes stay."""
        old = datetime.now(timezone.utc) - timedelta(hours=5)
        for key in ("face_jobs/old/input/rig/0/f_1000.jpg", "face_jobs/old/output/status.json"):
            self.s3.put_object(Bucket=BUCKET, Key=key, Body=b"x")
            self.s3.mtimes[(BUCKET, key)] = old
        # old photos but a fresh started.json: still running
        self.s3.put_object(Bucket=BUCKET, Key="face_jobs/running/input/f.jpg", Body=b"x")
        self.s3.mtimes[(BUCKET, "face_jobs/running/input/f.jpg")] = old
        self.s3.put_object(Bucket=BUCKET, Key="face_jobs/running/output/started.json", Body=b"{}")
        self.s3.put_object(Bucket=BUCKET, Key="face_jobs/mine/input/f.jpg", Body=b"x")
        self.s3.mtimes[(BUCKET, "face_jobs/mine/input/f.jpg")] = old
        self.s3.put_object(Bucket=BUCKET, Key="scans/f.jpg", Body=b"x")      # the scanner's own folder
        self.s3.mtimes[(BUCKET, "scans/f.jpg")] = old
        self.s3.put_object(Bucket=BUCKET, Key="jobs/obj/input/f.jpg", Body=b"x")   # object scan, same bucket
        self.s3.mtimes[(BUCKET, "jobs/obj/input/f.jpg")] = old

        deleted = r2.delete_stale_jobs(2 * 3600, BUCKET, client=self.s3, keep=["mine"])
        self.assertEqual(deleted, ["old"])
        keys = {k for (_, k) in self.s3.objects}
        self.assertFalse(any(k.startswith("face_jobs/old/") for k in keys))
        self.assertIn("face_jobs/running/input/f.jpg", keys)
        self.assertIn("face_jobs/mine/input/f.jpg", keys)
        self.assertIn("scans/f.jpg", keys)
        self.assertIn("jobs/obj/input/f.jpg", keys)

        # cloud_scan's default: every earlier job goes, the current one and other prefixes stay
        self.assertEqual(r2.delete_stale_jobs(0, BUCKET, client=self.s3, keep=["mine"], now=1e12), ["running"])
        keys = {k for (_, k) in self.s3.objects}
        self.assertEqual(keys, {"face_jobs/mine/input/f.jpg", "scans/f.jpg", "jobs/obj/input/f.jpg"})

    def test_keys(self):
        self.assertEqual(r2.input_key("j1", "rig\\0/f_1000.jpg"), "face_jobs/j1/input/rig/0/f_1000.jpg")
        self.assertEqual(r2.output_key("j1", "status.json"), "face_jobs/j1/output/status.json")
        for bad in ("", "a/b", ".."):
            with self.assertRaises(ValueError):
                r2.job_prefix(bad)

    def test_upload_download_roundtrip_keeps_rig_layout(self):
        src = self.tmp / "in"
        for cam in ("0", "1"):
            (src / "rig" / cam).mkdir(parents=True)
            (src / "rig" / cam / "f_1000.jpg").write_bytes(b"jpeg" + cam.encode())
            (src / "calib" / cam).mkdir(parents=True)
            (src / "calib" / cam / "f_1000.jpg").write_bytes(b"disc" + cam.encode())
        (src / "rig" / "0" / "empty.jpg").write_bytes(b"")          # failed capture: not uploaded
        res = r2.upload_job_images("j1", str(src), BUCKET, client=self.s3, workers=2)
        self.assertEqual(sorted(res["uploaded"]), [
            "face_jobs/j1/input/calib/0/f_1000.jpg", "face_jobs/j1/input/calib/1/f_1000.jpg",
            "face_jobs/j1/input/rig/0/f_1000.jpg", "face_jobs/j1/input/rig/1/f_1000.jpg"])
        self.assertEqual(res["failed"], ["face_jobs/j1/input/rig/0/empty.jpg"])

        expected = [r2.input_key("j1", "rig/0/f_1000.jpg"), r2.input_key("j1", "rig/1/f_1000.jpg")]
        self.assertEqual(r2.verify_keys(expected, BUCKET, prefix=r2.input_prefix("j1"), client=self.s3), [])

        dst = self.tmp / "pod_in"
        r2.download_job_images("j1", str(dst), BUCKET, client=self.s3)
        self.assertEqual((dst / "rig" / "1" / "f_1000.jpg").read_bytes(), b"jpeg1")
        self.assertEqual((dst / "calib" / "0" / "f_1000.jpg").read_bytes(), b"disc0")

    def test_delete_prefix_only_inside_jobs(self):
        self.s3.put_object(Bucket=BUCKET, Key="face_jobs/j1/input/a.jpg", Body=b"x")
        self.s3.put_object(Bucket=BUCKET, Key="face_jobs/j2/input/a.jpg", Body=b"x")
        self.assertEqual(r2.delete_prefix(r2.job_prefix("j1"), BUCKET, client=self.s3), 1)
        self.assertIn((BUCKET, "face_jobs/j2/input/a.jpg"), self.s3.objects)
        with self.assertRaises(r2.R2Error):
            r2.delete_prefix("", BUCKET, client=self.s3)

    def test_download_output_extracts_zip_and_rejects_escape(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("glb/high_model.glb", b"glb")
            zf.writestr("logs/run_log.txt", b"log")
        self.s3.put_object(Bucket=BUCKET, Key=r2.output_key("j1", "output.zip"), Body=buf.getvalue())
        r2.put_json({"state": "ok"}, r2.output_key("j1", "status.json"), BUCKET, client=self.s3)
        dest = r2.download_job_output("j1", str(self.tmp / "res" / "j1"), BUCKET, client=self.s3)
        self.assertEqual((dest / "glb" / "high_model.glb").read_bytes(), b"glb")
        self.assertEqual(json.loads((dest / "status.json").read_text())["state"], "ok")

        bad = io.BytesIO()
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("../evil.txt", b"x")
        self.s3.put_object(Bucket=BUCKET, Key=r2.output_key("j2", "output.zip"), Body=bad.getvalue())
        with self.assertRaises(r2.R2Error):
            r2.download_job_output("j2", str(self.tmp / "res" / "j2"), BUCKET, client=self.s3)
        self.assertFalse((self.tmp / "res" / "evil.txt").exists())

    def test_get_json_missing_is_none(self):
        self.assertIsNone(r2.get_json("face_jobs/x/output/status.json", BUCKET, client=self.s3))


class TestTempCredentials(unittest.TestCase):
    def test_request_is_object_level_and_prefix_scoped(self):
        body = r2.temp_credentials_request(BUCKET, "parent", "object-read-only", ["face_jobs/j1/input/"], 3600)
        self.assertEqual(body, {"bucket": BUCKET, "parentAccessKeyId": "parent", "permission": "object-read-only",
                                "prefixes": ["face_jobs/j1/input/"], "ttlSeconds": 3600})
        with self.assertRaises(ValueError):
            r2.temp_credentials_request(BUCKET, "parent", "admin-read-write", ["face_jobs/j1/"], 3600)
        with self.assertRaises(ValueError):
            r2.temp_credentials_request(BUCKET, "parent", "object-read-write", ["photos/"], 3600)
        with self.assertRaises(ValueError):
            r2.temp_credentials_request(BUCKET, "parent", "object-read-write", [], 3600)

    def test_pod_env_has_read_only_input_and_write_output(self):
        calls = []

        class Session:
            def post(self, url, headers, json, timeout):
                calls.append(json)
                resp = mock.Mock(status_code=200)
                n = len(calls)
                resp.json.return_value = {"success": True, "result": {
                    "accessKeyId": f"id{n}", "secretAccessKey": f"s{n}", "sessionToken": f"t{n}"}}
                return resp

        with mock.patch.object(r2, "requests", Session()), mock.patch.dict(os.environ, {
                "CLOUDFLARE_ACCOUNT_ID": "acc", "CLOUDFLARE_R2_API_TOKEN": "tok", "R2_ACCESS_KEY_ID": "parent",
                "R2_ENDPOINT_URL": "https://acc.r2.cloudflarestorage.com", "R2_BUCKET": BUCKET}):
            env = r2.pod_credentials_env("j1", 3600)
        self.assertEqual([c["permission"] for c in calls], ["object-read-only", "object-read-write"])
        self.assertEqual([c["prefixes"] for c in calls], [["face_jobs/j1/input/"], ["face_jobs/j1/output/"]])
        self.assertEqual(env["R2_IN_ACCESS_KEY_ID"], "id1")
        self.assertEqual(env["R2_OUT_SESSION_TOKEN"], "t2")
        self.assertNotIn("R2_SECRET_ACCESS_KEY", env)   # the account key never goes to the pod


class TestRunPod(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(rp.time, "sleep", lambda s: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def body(self, ram=16, compute_type="CPU"):
        return rp.build_pod_request("docker.io/u/face:v1", {"JOB_ID": "j1", "R2_OUT_SECRET_ACCESS_KEY": "x"},
                                    ram, 8, 30, ["NVIDIA RTX A4000", "NVIDIA GeForce RTX 3090"],
                                    compute_type=compute_type, cpu_flavors=["cpu5c", "cpu3g"])

    def test_cpu_request_sizes_vcpus_for_ram(self):
        b = self.body(ram=12)
        self.assertEqual((b["computeType"], b["cpuFlavorIds"], b["cpuFlavorPriority"]), ("CPU", ["cpu5c", "cpu3g"], "custom"))
        self.assertEqual(b["vcpuCount"], 8)               # min_vcpu covers 12 GB at 2 GB per vCPU
        self.assertEqual(self.body(ram=24)["vcpuCount"], 12)
        self.assertNotIn("gpuTypeIds", b)
        self.assertEqual(rp.redact(b)["env"]["R2_OUT_SECRET_ACCESS_KEY"], "***")
        self.assertEqual(rp.disk_gb_for(0.05), 11)
        with self.assertRaises(rp.RunPodError):
            self.body(compute_type="TPU")

    def test_gpu_request(self):
        b = self.body(compute_type="GPU")
        self.assertEqual(b["gpuTypeIds"], ["NVIDIA RTX A4000", "NVIDIA GeForce RTX 3090"])
        self.assertEqual(b["gpuTypePriority"], "custom")
        self.assertEqual((b["minRAMPerGPU"], b["minVCPUPerGPU"], b["gpuCount"]), (16, 8, 1))
        self.assertNotIn("cpuFlavorIds", b)

    def test_pod_terminated_after_status(self):
        api = FakeRunPod()
        statuses = iter([None, None, {"state": "ok"}])
        run = rp.run_pod(api, self.body(), lambda: next(statuses), poll_s=0)
        self.assertEqual((run.outcome, run.status["state"], run.terminated), ("finished", "ok", True))
        self.assertEqual(api.deleted, ["pod1"])

    def test_timeout_and_failed_terminate(self):
        api = FakeRunPod(delete_fails=True)
        run = rp.run_pod(api, self.body(), lambda: None, poll_s=0, max_runtime_s=-1)
        self.assertEqual(run.outcome, "timeout")
        self.assertFalse(run.terminated)

    def test_pod_that_never_starts_is_replaced(self):
        # Host stuck pulling the image: the pod stays RUNNING but never writes started.json
        api = FakeRunPod()
        started = iter([False, True])          # second pod starts at its first check
        statuses = iter([None, {"state": "ok"}])
        with mock.patch.object(rp, "BOOT_TIMEOUT_S", -1):
            runs = rp.run_job_pods(api, self.body, 16, lambda: next(statuses, None), read_started=lambda: next(started))
        self.assertEqual([r.outcome for r in runs], ["boot_timeout", "finished"])
        self.assertEqual(api.bodies[1]["vcpuCount"], 8)   # same request, new pod
        self.assertEqual(api.deleted, ["pod1", "pod2"])

    def test_started_pod_is_not_boot_timed_out(self):
        api = FakeRunPod()
        statuses = iter([None, None, {"state": "ok"}])
        run = rp.run_pod(api, self.body(), lambda: next(statuses), poll_s=0, read_started=lambda: True,
                         boot_timeout_s=-1)
        self.assertEqual(run.outcome, "finished")

    def test_create_retries_then_gives_up(self):
        api = FakeRunPod(fail_creates=99)
        self.assertIsNone(rp.run_pod(api, self.body(), lambda: None, poll_s=0))
        self.assertEqual(len(api.bodies), rp.MAX_CREATE_ATTEMPTS)
        self.assertEqual(api.created, [])

    def test_ram_insufficient_retries_with_more_ram(self):
        api = FakeRunPod()
        statuses = iter([{"state": "ram_insufficient", "required_gb": 30}, {"state": "ok"}])
        cleared = []
        runs = rp.run_job_pods(api, self.body, 16, lambda: next(statuses), before_retry=lambda: cleared.append(1))
        self.assertEqual([r.status["state"] for r in runs], ["ram_insufficient", "ok"])
        self.assertEqual(api.bodies[1]["vcpuCount"], 18)  # 30 GB measured * 1.2 = 36 GB at 2 GB per vCPU
        self.assertEqual(cleared, [1])
        self.assertEqual(api.deleted, ["pod1", "pod2"])

    def test_pipeline_error_is_not_retried(self):
        api = FakeRunPod()
        runs = rp.run_job_pods(api, self.body, 16, lambda: {"state": "pipeline_error"})
        self.assertEqual(len(runs), 1)

    def test_exit_without_status_retries_with_more_ram(self):
        api = FakeRunPod(exit_after=1)
        runs = rp.run_job_pods(api, self.body, 16, lambda: None)
        self.assertEqual([r.outcome for r in runs], ["exited", "exited"])
        self.assertEqual(api.bodies[1]["vcpuCount"], 12)  # 16 GB * 1.5


class TestPodJob(unittest.TestCase):
    def test_upload_outputs_writes_status_last(self):
        import runpod_job
        s3 = FakeS3()
        order = []
        real_put = s3.put_object

        def tracking_upload(file_name, bucket, key):
            order.append(key)
            FakeS3.upload_file(s3, file_name, bucket, key)

        def tracking_put(**kw):
            order.append(kw["Key"])
            real_put(**kw)

        s3.upload_file, s3.put_object = tracking_upload, tracking_put
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for name in ("output.zip", "stats.json", "run_log.txt"):
                (out / name).write_bytes(b"data")
            with mock.patch.object(runpod_job, "OUTPUT_DIR", out):
                runpod_job.upload_outputs(s3, BUCKET, "j1", {"state": "ok"})
        self.assertEqual(order[-1], "face_jobs/j1/output/status.json")
        self.assertEqual(len(order), 4)


def _face_job_input(s3, job_id, tmp, with_rig=True, with_calib=True):
    from PIL import Image
    src = tmp / "src"
    for folder, ok in (("rig", with_rig), ("calib", with_calib)):
        if not ok:
            continue
        for cam in ("0", "1"):
            names = ["f_150.jpg"] if folder == "rig" else ["f_1000.jpg", "f_1001.jpg"]
            for name in names:
                p = src / folder / cam / name
                p.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.full((304, 405, 3), 120, np.uint8)).save(p)
                s3.upload_file(str(p), BUCKET, r2.input_key(job_id, f"{folder}/{cam}/{name}"))


class TestPodJobEndToEnd(unittest.TestCase):
    def run_pod(self, s3, job_id, tmp, pipeline):
        import runpod_job
        import run_photogrammetry_runpod
        env = {"JOB_ID": job_id, "R2_BUCKET": BUCKET, "POD_MAX_RUNTIME_S": "3600",
               "JOB_JSON": json.dumps({"spectacles": False, "unknown_option": 1})}
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(r2, "client_from_env", lambda prefix="R2_": s3), \
                mock.patch.object(runpod_job, "INPUT_DIR", tmp / "in"), \
                mock.patch.object(runpod_job, "OUTPUT_DIR", tmp / "out"), \
                mock.patch.object(run_photogrammetry_runpod, "run_photogrammetry_pipeline", pipeline):
            code = runpod_job.main()
            status = r2.get_json(r2.output_key(job_id, "status.json"), BUCKET, client=s3)
            again = runpod_job.main()   # container restarted by RunPod: must not run the job twice
        for h in list(__import__("logging").getLogger().handlers):
            if getattr(h, "baseFilename", "").startswith(str(tmp)):
                __import__("logging").getLogger().removeHandler(h)
                h.close()
        return code, status, again

    def test_pod_downloads_runs_pipeline_and_reports(self):
        """runpod_job.main(): photos from (fake) R2 -> pipeline on rig/ (calib/ next to it) -> output.zip + status ok."""
        s3 = FakeS3()
        calls = []

        def pipeline(**kw):
            calls.append(kw)
            out = Path(kw["output_dir"])
            (out / "glb").mkdir(parents=True, exist_ok=True)
            (out / "glb" / "high_model.glb").write_bytes(b"glb")
            (out / "photogrammetry_manifest.json").write_text(json.dumps({"duration_s": 1.0}))
            return {"status": "success", "calibration": {"source": "turntable_poses"}}

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _face_job_input(s3, "j9", tmp)
            code, status, again = self.run_pod(s3, "j9", tmp, pipeline)
            self.assertTrue((tmp / "in" / "calib" / "1" / "f_1001.jpg").is_file())
        self.assertEqual(code, 0, status)
        self.assertEqual((status["state"], status["calibration"]), ("ok", "turntable_poses"))
        self.assertEqual(again, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["spectacles"], False)
        self.assertNotIn("unknown_option", calls[0])
        self.assertEqual(Path(calls[0]["rig_dir"]).name, "rig")
        self.assertEqual(calls[0]["reconstruction_mode"], "dense")
        names = zipfile.ZipFile(io.BytesIO(s3.objects[(BUCKET, r2.output_key("j9", "output.zip"))])).namelist()
        self.assertIn("logs/run_log.txt", names)
        self.assertIn("stats.json", names)
        self.assertIn("glb/high_model.glb", names)
        self.assertIn((BUCKET, r2.output_key("j9", "run_log.txt")), s3.objects)
        self.assertIn((BUCKET, r2.output_key("j9", "started.json")), s3.objects)   # boot signal for the client
        stats = json.loads(s3.objects[(BUCKET, r2.output_key("j9", "stats.json"))])
        self.assertEqual(stats["pod"]["state"], "ok")
        self.assertIn("pipeline", stats["pod"]["timings_s"])

    def test_pipeline_failure_and_missing_face_pair_are_reported(self):
        s3 = FakeS3()

        def failing(**kw):
            raise RuntimeError("turntable_poses.py failed (exit 1): no tags")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _face_job_input(s3, "j1", tmp)
            code, status, _ = self.run_pod(s3, "j1", tmp, failing)
        self.assertEqual((code, status["state"]), (1, "pipeline_error"))
        self.assertIn("no tags", status["error"])
        self.assertNotIn((BUCKET, r2.output_key("j1", "output.zip")), s3.objects)
        self.assertIn((BUCKET, r2.output_key("j1", "run_log.txt")), s3.objects)

        s3 = FakeS3()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _face_job_input(s3, "j2", tmp, with_rig=False)
            code, status, _ = self.run_pod(s3, "j2", tmp, failing)
        self.assertEqual((code, status["state"]), (1, "input_error"))


class TestPlanJob(unittest.TestCase):
    def test_plan_from_photos(self):
        from cloud.job import plan_job
        s3 = FakeS3()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _face_job_input(s3, "j", tmp)
            plan = plan_job(str(tmp / "src"))
            self.assertEqual((plan["n_face_photos"], plan["n_calib_photos"], plan["width"], plan["height"]), (2, 4, 405, 304))
            self.assertGreaterEqual(plan["min_ram_gb"], plan["ram_estimate_gb"])
            shutil.rmtree(tmp / "src" / "rig")
            with self.assertRaises(r2.R2Error):
                plan_job(str(tmp / "src"))


class TestCloudScanTool(unittest.TestCase):
    def setUp(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
        import cloud_scan
        self.tool = cloud_scan

    def test_set_values(self):
        opts = self.tool.parse_set(["tag_scale=1.002", "spectacles=false", "--name=alin"])
        self.assertEqual(opts, {"tag_scale": 1.002, "spectacles": False, "name": "alin"})
        with self.assertRaises(SystemExit):
            self.tool.parse_set(["output_dir=/x"])
        with self.assertRaises(SystemExit):
            self.tool.parse_set(["tag_scale"])
        with self.assertRaises(SystemExit):
            self.tool.parse_set(["tag_scal=1.0"])             # typo: refused, not silently ignored in the pod
        self.assertEqual(self.tool.parse_set(["no_spectacles"]), {"spectacles": False})

    def test_known_options_are_job_config_keys(self):
        """The pod accepts every JobConfig key except the local paths; cloud_scan reserves exactly those."""
        from dataclasses import fields
        import inspect
        import run_photogrammetry_runpod as rp_mod
        keys = {f.name for f in fields(rp_mod.JobConfig)}
        self.assertEqual(self.tool.PIPELINE_OPTIONS, keys - rp_mod.PATH_OPTIONS)
        self.assertEqual(self.tool.RESERVED_OPTIONS - {"job_id"}, rp_mod.PATH_OPTIONS)
        self.assertEqual(keys, set(inspect.signature(rp_mod.run_photogrammetry_pipeline).parameters))
        with self.assertRaises(ValueError):
            rp_mod.JobConfig.from_dict({"tag_scal": 1.0})

    def test_ctrl_c_cancels_the_job(self):
        seen = {}

        def fake_job(cancel_event=None, **kwargs):
            seen["cancelled"] = cancel_event.wait(timeout=5)
            return {"job_id": "x", "timings_s": {}, "result": "failed"}

        joins = iter([KeyboardInterrupt()])

        real_join = self.tool.threading.Thread.join

        def join(thread, timeout=None):
            err = next(joins, None)
            if err:
                raise err
            return real_join(thread, timeout)

        with mock.patch.object(self.tool, "run_cloud_job", fake_job), \
                mock.patch.object(self.tool.threading.Thread, "join", join):
            report = self.tool.run_interruptible(input_dir="x")
        self.assertTrue(seen["cancelled"])
        self.assertEqual(report["result"], "failed")


if __name__ == "__main__":
    unittest.main()
