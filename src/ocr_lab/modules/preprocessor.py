from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from PIL import Image, ImageEnhance, ImageOps


class PassthroughPreprocessor:
    """Default no-op preprocessor."""

    def __init__(self, **_: Any) -> None:
        pass

    def preprocess(self, image_path: Path, output_dir: Path | None = None) -> Path:
        return image_path


class ResizePreprocessor:
    """Standard image resizer maintaining aspect ratio."""

    def __init__(self, max_size: int = 1536, **_: Any) -> None:
        self.max_size = int(max_size)

    def preprocess(self, image_path: Path, output_dir: Path | None = None) -> Path:
        if output_dir is None:
            return image_path
        output_dir.mkdir(parents=True, exist_ok=True)
        dest_path = output_dir / image_path.name

        with Image.open(image_path) as img:
            w, h = img.size
            if max(w, h) <= self.max_size:
                return image_path
            scale = self.max_size / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            resized = img.resize(new_size, Image.Resampling.LANCZOS)
            resized.save(dest_path)

        # Copy sidecar if mock is being tested
        sidecar = image_path.with_suffix(image_path.suffix + ".ocr.json")
        if sidecar.exists():
            shutil.copy2(sidecar, dest_path.with_suffix(dest_path.suffix + ".ocr.json"))

        return dest_path


class GrayscaleContrastPreprocessor:
    """Grayscale conversion and contrast enhancement for difficult OCR scans."""

    def __init__(self, contrast_factor: float = 1.5, **_: Any) -> None:
        self.contrast_factor = float(contrast_factor)

    def preprocess(self, image_path: Path, output_dir: Path | None = None) -> Path:
        if output_dir is None:
            return image_path
        output_dir.mkdir(parents=True, exist_ok=True)
        dest_path = output_dir / image_path.name

        with Image.open(image_path) as img:
            gray = ImageOps.grayscale(img)
            enhancer = ImageEnhance.Contrast(gray)
            enhanced = enhancer.enhance(self.contrast_factor)
            enhanced.save(dest_path)

        sidecar = image_path.with_suffix(image_path.suffix + ".ocr.json")
        if sidecar.exists():
            shutil.copy2(sidecar, dest_path.with_suffix(dest_path.suffix + ".ocr.json"))

        return dest_path


class AdaptiveContrastPreprocessor:
    """Adaptive local contrast enhancement (CLAHE-like) and unsharp masking for dot-matrix stamps."""

    def __init__(self, cutoff: float = 2.0, unsharp: bool = True, **_: Any) -> None:
        self.cutoff = float(cutoff)
        self.unsharp = bool(unsharp)

    def preprocess(self, image_path: Path, output_dir: Path | None = None) -> Path:
        if output_dir is None:
            return image_path
        output_dir.mkdir(parents=True, exist_ok=True)
        dest_path = output_dir / image_path.name

        with Image.open(image_path) as img:
            gray = ImageOps.grayscale(img)
            autocontrast = ImageOps.autocontrast(gray, cutoff=self.cutoff)
            if self.unsharp:
                from PIL import ImageFilter
                enhanced = autocontrast.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            else:
                enhanced = autocontrast
            enhanced.save(dest_path)

        sidecar = image_path.with_suffix(image_path.suffix + ".ocr.json")
        if sidecar.exists():
            shutil.copy2(sidecar, dest_path.with_suffix(dest_path.suffix + ".ocr.json"))

        return dest_path


class LocalClahePreprocessor:
    """Local CLAHE retry for reflective or dot-matrix text."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: int = 8, **_: Any) -> None:
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = int(tile_grid_size)

    def preprocess(self, image_path: Path, output_dir: Path | None = None) -> Path:
        if output_dir is None:
            return image_path
        output_dir.mkdir(parents=True, exist_ok=True)
        dest_path = output_dir / image_path.name
        import cv2
        import numpy as np

        with Image.open(image_path) as img:
            gray = np.asarray(ImageOps.grayscale(img))
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.tile_grid_size, self.tile_grid_size))
        enhanced = clahe.apply(gray)
        # Keep the retry cheap: full-image detailEnhance is prohibitively slow
        # on 12MP photos and is not part of the intended CPU baseline.
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        sharpened = cv2.addWeighted(enhanced, 1.15, blurred, -0.15, 0)
        Image.fromarray(sharpened).save(dest_path)
        return dest_path
