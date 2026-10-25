"""
Facial Scanner & Landmark Extractor using Google MediaPipe & Computational Graphics
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarksConnections, drawing_styles, drawing_utils


# Pre-extract canonical triangles (852 triangles) from MediaPipe tessellation
def _build_tessellation_triangles() -> List[Tuple[int, int, int]]:
    tess = list(FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION)
    triangles = []
    for i in range(0, len(tess), 3):
        e1, e2, e3 = tess[i], tess[i + 1], tess[i + 2]
        nodes = {e1.start, e1.end, e2.start, e2.end, e3.start, e3.end}
        if len(nodes) == 3:
            triangles.append((e1.start, e2.start, e3.start))
    return triangles


TESSELLATION_TRIANGLES = _build_tessellation_triangles()

# Key landmark indices in the canonical 468/478 face mesh
KEY_LANDMARK_INDICES = {
    "nose_tip": 1,
    "nose_bottom": 2,
    "chin": 152,
    "forehead": 10,
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_outer": 263,
    "right_eye_inner": 362,
    "mouth_left_corner": 61,
    "mouth_right_corner": 291,
    "upper_lip_center": 0,
    "lower_lip_center": 17,
    "left_iris_center": 468,
    "right_iris_center": 473,
}


class FaceScanner:
    """
    High-performance facial landmark detector and 3D mesh extractor.
    """

    def __init__(
        self,
        model_path: str = "assets/face_landmarker.task",
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        output_face_blendshapes: bool = True,
        output_facial_transformation_matrixes: bool = True,
    ):
        p = Path(model_path)
        if not p.is_file():
            asset_candidate = Path(__file__).parent / "assets" / p.name
            if asset_candidate.is_file():
                p = asset_candidate
            else:
                root_candidate = Path(__file__).parent / p.name
                if root_candidate.is_file():
                    p = root_candidate
        self.model_path = str(p.resolve())
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file '{self.model_path}' not found. Please provide a valid face_landmarker.task model."
            )

        self.num_faces = num_faces
        self.output_face_blendshapes = output_face_blendshapes
        self.output_facial_transformation_matrixes = output_facial_transformation_matrixes

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        self.options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=self.num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=self.output_face_blendshapes,
            output_facial_transformation_matrixes=self.output_facial_transformation_matrixes,
        )
        self.detector = vision.FaceLandmarker.create_from_options(self.options)

    def close(self):
        """Cleanly releases resources."""
        if hasattr(self, "detector") and self.detector is not None:
            self.detector.close()
            self.detector = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def scan_image(
        self,
        image_input: Union[str, Path, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Processes an image and extracts 478 face landmarks, blendshapes,
        bounding boxes, and transformation matrices.

        :param image_input: Filepath or BGR numpy array
        :return: Dict containing detected faces and structured landmark coordinates
        """
        # Load image
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            img_bgr = cv2.imread(path_str)
            if img_bgr is None:
                raise ValueError(f"Could not read image from path: {path_str}")
            source_name = Path(path_str).name
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input
            source_name = "in_memory_array"
        else:
            raise TypeError("image_input must be a file path or numpy array.")

        height, width = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # Run inference
        detection_result = self.detector.detect(mp_image)

        faces_data = []
        for face_idx, landmarks in enumerate(detection_result.face_landmarks):
            # 1. Normalized & Pixel coordinates
            norm_points = []
            pixel_points = []
            xs_px = []
            ys_px = []

            for i, lm in enumerate(landmarks):
                # Normalized [0..1]
                norm_points.append({
                    "index": i,
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                })
                # Pixel space [0..W, 0..H]
                # MediaPipe z represents depth relative to face center, scaled with width
                px = float(lm.x * width)
                py = float(lm.y * height)
                pz = float(lm.z * width)
                pixel_points.append({
                    "index": i,
                    "x": px,
                    "y": py,
                    "z": pz,
                })
                xs_px.append(px)
                ys_px.append(py)

            # Bounding box [xmin, ymin, xmax, ymax]
            bbox = {
                "xmin": max(0.0, float(min(xs_px))),
                "ymin": max(0.0, float(min(ys_px))),
                "xmax": min(float(width), float(max(xs_px))),
                "ymax": min(float(height), float(max(ys_px))),
            }

            # 2. Key landmarks mapping
            key_points = {}
            for name, idx in KEY_LANDMARK_INDICES.items():
                if idx < len(pixel_points):
                    key_points[name] = {
                        "index": idx,
                        "pixel": [pixel_points[idx]["x"], pixel_points[idx]["y"], pixel_points[idx]["z"]],
                        "normalized": [norm_points[idx]["x"], norm_points[idx]["y"], norm_points[idx]["z"]],
                    }

            # 3. Blendshapes (ARKit-compatible 52 weights)
            blendshapes = {}
            if detection_result.face_blendshapes and face_idx < len(detection_result.face_blendshapes):
                for b in detection_result.face_blendshapes[face_idx]:
                    blendshapes[b.category_name] = float(b.score)

            # 4. Facial transformation matrix (4x4)
            transform_matrix = None
            if detection_result.facial_transformation_matrixes and face_idx < len(
                detection_result.facial_transformation_matrixes
            ):
                transform_matrix = [
                    [float(val) for val in row]
                    for row in detection_result.facial_transformation_matrixes[face_idx]
                ]

            faces_data.append({
                "face_index": face_idx,
                "total_landmarks": len(landmarks),
                "bounding_box": bbox,
                "key_points": key_points,
                "normalized_landmarks": norm_points,
                "pixel_landmarks": pixel_points,
                "blendshapes": blendshapes,
                "transformation_matrix": transform_matrix,
            })

        return {
            "source": source_name,
            "image_dimensions": {"width": width, "height": height, "channels": img_bgr.shape[2] if len(img_bgr.shape) > 2 else 1},
            "num_faces_detected": len(faces_data),
            "faces": faces_data,
            "_raw_detection_result": detection_result,
            "_original_bgr": img_bgr,
        }

    @staticmethod
    def draw_annotations(
        scan_result: Dict[str, Any],
        draw_tesselation: bool = True,
        draw_contours: bool = True,
        draw_irises: bool = True,
        draw_key_points: bool = True,
    ) -> np.ndarray:
        """
        Renders face mesh, contours, and irises onto a copy of the original image.
        """
        annotated = scan_result["_original_bgr"].copy()
        raw_result = scan_result.get("_raw_detection_result")
        if raw_result is None or not raw_result.face_landmarks:
            return annotated

        for face_landmarks in raw_result.face_landmarks:
            if draw_tesselation:
                drawing_utils.draw_landmarks(
                    image=annotated,
                    landmark_list=face_landmarks,
                    connections=FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
                )
            if draw_contours:
                drawing_utils.draw_landmarks(
                    image=annotated,
                    landmark_list=face_landmarks,
                    connections=FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
                )
            if draw_irises:
                for iris_conn in [
                    FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                    FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                ]:
                    drawing_utils.draw_landmarks(
                        image=annotated,
                        landmark_list=face_landmarks,
                        connections=iris_conn,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
                    )

        if draw_key_points:
            for face in scan_result["faces"]:
                for name, kp in face["key_points"].items():
                    px, py, _ = kp["pixel"]
                    cv2.circle(annotated, (int(px), int(py)), 3, (0, 0, 255), -1)

        return annotated

    @staticmethod
    def export_obj(
        face_data: Dict[str, Any],
        output_path: Union[str, Path],
        coord_space: str = "normalized",
    ) -> str:
        """
        Exports a 3D Wavefront .OBJ mesh file using canonical MediaPipe triangulation (852 triangles).

        :param face_data: Single face dictionary from scan_result["faces"]
        :param output_path: Destination .obj file path
        :param coord_space: 'normalized' (x in [0,1], y inverted for 3D) or 'pixel' (x in [0,W], y in [0,H])
        :return: Path to saved OBJ file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        landmarks = (
            face_data["normalized_landmarks"]
            if coord_space == "normalized"
            else face_data["pixel_landmarks"]
        )

        lines = [
            "# Wavefront OBJ Face Mesh generated by MediaPipe Face Scanner",
            f"# Total vertices: {len(landmarks)}",
            f"# Total triangles: {len(TESSELLATION_TRIANGLES)}",
            f"# Coordinate space: {coord_space}",
            "",
        ]

        # In standard 3D graphics (OpenGL/Blender), Y goes UP, while in 2D image coordinates Y goes DOWN.
        # We invert Y so the 3D mesh is upright.
        for lm in landmarks:
            vx = lm["x"]
            vy = -lm["y"] if coord_space == "normalized" else -lm["y"]
            vz = -lm["z"]  # Z points into the screen
            lines.append(f"v {vx:.6f} {vy:.6f} {vz:.6f}")

        lines.append("")
        lines.append("# Triangular Faces (1-indexed)")
        for v1, v2, v3 in TESSELLATION_TRIANGLES:
            # OBJ uses 1-based indexing
            lines.append(f"f {v1 + 1} {v2 + 1} {v3 + 1}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        return str(output_path)

    @staticmethod
    def export_json(
        scan_result: Dict[str, Any],
        output_path: Union[str, Path],
        indent: int = 2,
    ) -> str:
        """
        Exports structured scan results to a clean JSON file (without internal raw objects).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Filter out internal OpenCV/MediaPipe raw references
        export_dict = {
            "source": scan_result["source"],
            "image_dimensions": scan_result["image_dimensions"],
            "num_faces_detected": scan_result["num_faces_detected"],
            "faces": scan_result["faces"],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_dict, f, indent=indent)

        return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="MediaPipe Face Scanner & Landmark Extractor (478 Landmarks & 3D Mesh)"
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to input image file or directory of images"
    )
    parser.add_argument(
        "-o", "--output-dir", default="./output", help="Directory where results will be saved"
    )
    parser.add_argument(
        "-m", "--model", default="face_landmarker.task", help="Path to face_landmarker.task model"
    )
    parser.add_argument(
        "--num-faces", type=int, default=1, help="Maximum number of faces to detect per image"
    )
    parser.add_argument(
        "--save-json", action="store_true", default=True, help="Save structured landmarks JSON"
    )
    parser.add_argument(
        "--save-annotated", action="store_true", default=True, help="Save annotated image with mesh overlay"
    )
    parser.add_argument(
        "--save-obj", action="store_true", default=True, help="Save 3D .OBJ facial mesh"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect image files
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = [p for p in input_path.iterdir() if p.suffix.lower() in valid_exts]
        image_files.sort()
    else:
        print(f"Error: Invalid input path '{input_path}'")
        sys.exit(1)

    if not image_files:
        print(f"No valid images found in {input_path}")
        sys.exit(0)

    print(f"Initializing FaceScanner with model: {args.model}...")
    with FaceScanner(model_path=args.model, num_faces=args.num_faces) as scanner:
        print(f"Processing {len(image_files)} image(s)...")

        for img_p in image_files:
            print(f"\nScanning: {img_p.name}...")
            result = scanner.scan_image(img_p)
            num_faces = result["num_faces_detected"]
            print(f"  -> Detected {num_faces} face(s).")

            stem = img_p.stem

            if args.save_json:
                json_p = output_dir / f"{stem}_landmarks.json"
                scanner.export_json(result, json_p)
                print(f"  -> Saved JSON: {json_p}")

            if args.save_annotated and num_faces > 0:
                annotated_img = scanner.draw_annotations(result)
                img_out_p = output_dir / f"{stem}_annotated.jpg"
                cv2.imwrite(str(img_out_p), annotated_img)
                print(f"  -> Saved Visualization: {img_out_p}")

            if args.save_obj and num_faces > 0:
                for face in result["faces"]:
                    face_idx = face["face_index"]
                    obj_p = output_dir / f"{stem}_face{face_idx}.obj"
                    scanner.export_obj(face, obj_p, coord_space="normalized")
                    print(f"  -> Saved 3D OBJ Mesh: {obj_p}")

    print("\nProcessing complete! All outputs saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
