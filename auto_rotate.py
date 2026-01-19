import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
ROTATE_STRATEGIES = ("auto", "pca", "minrect", "top-edge", "hough")
AUTO_STRATEGY_ORDER = ("top-edge", "hough", "pca", "minrect")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Auto-rotate RGBA (or RGB + mask) images by estimating the subject "
            "tilt angle from mask contours (multiple strategies supported)."
        )
    )
    parser.add_argument("--input", required=True, help="Input file or directory.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument(
        "--output-ext",
        default=".png",
        help="Output file extension, default: .png",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into input directory.",
    )
    parser.add_argument(
        "--mask-dir",
        default=None,
        help="Directory with masks (for RGB inputs).",
    )
    parser.add_argument(
        "--mask-suffix",
        default="",
        help="Mask filename suffix before extension, default: empty",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Alpha threshold (0-255) for mask binarization, default: 10",
    )
    parser.add_argument(
        "--close-kernel",
        type=int,
        default=3,
        help="Morph close kernel size (odd int). 0 to disable, default: 3",
    )
    parser.add_argument(
        "--close-iter",
        type=int,
        default=1,
        help="Morph close iterations, default: 1",
    )
    parser.add_argument(
        "--min-angle",
        type=float,
        default=0.5,
        help="Min absolute angle to rotate, default: 0.5",
    )
    parser.add_argument(
        "--max-angle",
        type=float,
        default=25.0,
        help="Mark as suspicious if abs(angle) exceeds, default: 25.0",
    )
    parser.add_argument(
        "--min-aspect",
        type=float,
        default=1.2,
        help="Min long/short ratio, default: 1.2",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.005,
        help="Min area ratio for valid mask, default: 0.005",
    )
    parser.add_argument(
        "--rotate-strategy",
        default="auto",
        choices=ROTATE_STRATEGIES,
        help="Rotation strategy: auto/pca/minrect/top-edge/hough, default: auto",
    )
    parser.add_argument(
        "--white",
        action="store_true",
        help="Composite onto white background output.",
    )
    parser.add_argument(
        "--no-rmbg",
        action="store_true",
        help="Disable Inspyrenet RMBG fallback for RGB inputs.",
    )
    parser.add_argument(
        "--rmbg-model",
        default="rmbg-model/inspyrenet.safetensors",
        help="Path to Inspyrenet safetensors model.",
    )
    parser.add_argument(
        "--rmbg-base-size",
        nargs=2,
        type=int,
        default=(1024, 1024),
        metavar=("W", "H"),
        help="Inspyrenet base resize, default: 1024 1024",
    )
    parser.add_argument(
        "--rmbg-device",
        default=None,
        help="RMBG device override (cpu/cuda:0). Default: auto",
    )
    return parser.parse_args()


def collect_images(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        return []
    if recursive:
        return [p for p in input_path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]
    return [p for p in input_path.iterdir() if p.suffix.lower() in SUPPORTED_EXTS]


def find_mask_path(img_path: Path, mask_dir: Path | None, mask_suffix: str) -> Path | None:
    if not mask_dir:
        return None
    candidates = [
        mask_dir / f"{img_path.stem}{mask_suffix}{img_path.suffix}",
        mask_dir / f"{img_path.stem}{mask_suffix}.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


class InspyrenetRunner:
    def __init__(
        self,
        model_path: Path,
        base_size: tuple[int, int],
        device: str | None,
    ):
        self.model_path = model_path
        self.base_size = base_size
        self.device = device
        self._loaded = False

    def _load(self):
        try:
            import torch
            import torchvision.transforms as transforms
            from safetensors.torch import load_file
            from transparent_background.InSPyReNet import InSPyReNet_SwinB
            from transparent_background.utils import normalize, static_resize, tonumpy, totensor
        except Exception as exc:
            raise RuntimeError(
                "Missing RMBG dependencies. Ensure transparent-background and "
                "safetensors are installed."
            ) from exc

        if not self.model_path.exists():
            raise FileNotFoundError(f"Inspyrenet model not found: {self.model_path}")

        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self._torch = torch
        self._F = torch.nn.functional
        self._transform = transforms.Compose(
            [
                static_resize(list(self.base_size)),
                tonumpy(),
                normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                totensor(),
            ]
        )

        state = load_file(str(self.model_path))
        model = InSPyReNet_SwinB(
            depth=64,
            pretrained=False,
            base_size=list(self.base_size),
            threshold=None,
        )
        model.load_state_dict(state, strict=True)
        model.eval()
        model = model.to(self.device)
        self._model = model
        self._loaded = True

    def predict_alpha(self, img: Image.Image) -> Image.Image:
        if not self._loaded:
            self._load()

        img_rgb = img.convert("RGB")
        x = self._transform(img_rgb).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            pred = self._model(x)
        pred = self._F.interpolate(
            pred,
            size=img_rgb.size[::-1],
            mode="bilinear",
            align_corners=True,
        )
        pred = pred[0, 0]
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        alpha = (pred * 255.0).clamp(0, 255).byte().cpu().numpy()
        return Image.fromarray(alpha, mode="L")


def load_image_and_alpha(
    img_path: Path,
    mask_path: Path | None,
    rmbg_runner: InspyrenetRunner | None,
):
    img = Image.open(img_path)
    img.load()

    has_alpha = img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    )
    img_rgba = img.convert("RGBA") if img.mode != "RGBA" else img

    if has_alpha:
        alpha_img = img_rgba.split()[-1]
        return img_rgba, alpha_img

    if mask_path:
        mask_img = Image.open(mask_path).convert("L")
        if mask_img.size != img_rgba.size:
            mask_img = mask_img.resize(img_rgba.size, Image.NEAREST)
        return img_rgba, mask_img

    if rmbg_runner is not None:
        alpha_img = rmbg_runner.predict_alpha(img)
        return img_rgba, alpha_img

    return img_rgba, None


def normalize_long_angle(angle: float, w: float, h: float) -> float:
    if w >= h:
        theta_long = angle
    else:
        theta_long = angle + 90.0

    if theta_long <= -90.0:
        theta_long += 180.0
    if theta_long > 90.0:
        theta_long -= 180.0

    if theta_long > 45.0:
        theta_long -= 90.0
    if theta_long < -45.0:
        theta_long += 90.0

    return theta_long


def normalize_angle_180(angle: float) -> float:
    if angle <= -90.0:
        angle += 180.0
    if angle > 90.0:
        angle -= 180.0
    return angle


def normalize_axis_angle(angle: float) -> float:
    angle = normalize_angle_180(angle)
    if angle > 45.0:
        angle -= 90.0
    if angle < -45.0:
        angle += 90.0
    return angle


def get_largest_contour(mask_bin: np.ndarray) -> np.ndarray | None:
    mask_u8 = (mask_bin.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def robust_line_fit(
    xs: np.ndarray,
    ys: np.ndarray,
    min_points: int,
    max_iter: int = 3,
    sigma: float = 2.5,
) -> dict | None:
    if xs.size < min_points:
        return None
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    inliers = np.ones_like(x, dtype=bool)
    sigma_est = None

    for _ in range(max_iter):
        if inliers.sum() < min_points:
            return None
        m, b = np.polyfit(x[inliers], y[inliers], 1)
        residuals = y - (m * x + b)
        res = residuals[inliers]
        med = np.median(res)
        mad = np.median(np.abs(res - med))
        if mad < 1e-6:
            sigma_est = 0.0
            break
        sigma_est = 1.4826 * mad
        new_inliers = np.abs(residuals) <= sigma * sigma_est
        if new_inliers.sum() == inliers.sum():
            break
        inliers = new_inliers

    if inliers.sum() < min_points:
        return None
    if sigma_est is None:
        res = residuals[inliers]
        med = np.median(res)
        mad = np.median(np.abs(res - med))
        sigma_est = 1.4826 * mad

    return {
        "m": float(m),
        "b": float(b),
        "inlier_ratio": float(inliers.sum() / max(1, x.size)),
        "sigma_est": float(sigma_est),
    }


def estimate_top_edge_angle(
    mask_bin: np.ndarray,
    bbox: tuple[int, int, int, int],
    min_points: int,
    top_band_ratio: float = 0.45,
) -> dict | None:
    x0, y0, w, h = bbox
    if w <= 0 or h <= 0:
        return None

    top_limit = y0 + max(1, int(round(h * top_band_ratio)))
    xs = []
    ys = []
    for x in range(x0, x0 + w):
        col = mask_bin[y0 : top_limit + 1, x]
        if not np.any(col):
            continue
        y = y0 + int(np.argmax(col))
        xs.append(x)
        ys.append(y)

    if len(xs) < min_points:
        return None

    xs_arr = np.array(xs, dtype=np.float64)
    ys_arr = np.array(ys, dtype=np.float64)
    coverage_ratio = float(len(xs) / max(1, w))
    x_span = float(xs_arr.max() - xs_arr.min())
    x_span_ratio = float(x_span / max(1.0, float(w)))
    if coverage_ratio < 0.4 or x_span_ratio < 0.5:
        return None

    fit = robust_line_fit(xs_arr, ys_arr, min_points=min_points)
    if fit is None:
        return None

    sigma_norm = float(fit["sigma_est"] / max(1.0, float(h)))
    if fit["inlier_ratio"] < 0.6 or sigma_norm > 0.03:
        return None

    theta_line = math.degrees(math.atan(fit["m"]))
    theta_line = normalize_angle_180(theta_line)
    angle = theta_line

    return {
        "angle": float(angle),
        "theta_line": float(theta_line),
        "inlier_ratio": float(fit["inlier_ratio"]),
        "coverage_ratio": coverage_ratio,
        "x_span_ratio": x_span_ratio,
        "sigma_norm": sigma_norm,
    }


def estimate_pca_angle(
    mask_bin: np.ndarray,
    min_points: int = 200,
    max_points: int = 200_000,
) -> dict | None:
    ys, xs = np.where(mask_bin)
    if xs.size < min_points:
        return None

    if xs.size > max_points:
        step = max(1, int(xs.size // max_points))
        xs = xs[::step]
        ys = ys[::step]

    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    xs -= xs.mean()
    ys -= ys.mean()
    cov = np.cov(np.stack([xs, ys], axis=0))
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, int(np.argmax(vals))]
    theta = math.degrees(math.atan2(v[1], v[0]))
    theta = normalize_axis_angle(theta)
    return {
        "angle": float(theta),
    }


def estimate_hough_angle(
    mask_bin: np.ndarray,
    bbox: tuple[int, int, int, int],
    max_abs_angle: float,
) -> dict | None:
    x0, y0, bw, bh = bbox
    if bw <= 0 or bh <= 0:
        return None

    mask_u8 = (mask_bin.astype(np.uint8) * 255)
    edges = cv2.Canny(mask_u8, 50, 150)
    min_line_len = max(30, int(bw * 0.35))
    max_line_gap = max(10, int(bw * 0.05))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=50,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return None

    angles = []
    weights = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < min_line_len:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        angle = normalize_angle_180(angle)
        if abs(angle) > max_abs_angle:
            continue
        angles.append(angle)
        weights.append(length)

    if not angles:
        return None

    angles_arr = np.array(angles, dtype=np.float64)
    weights_arr = np.array(weights, dtype=np.float64)
    order = np.argsort(angles_arr)
    angles_arr = angles_arr[order]
    weights_arr = weights_arr[order]
    cum = np.cumsum(weights_arr)
    cutoff = weights_arr.sum() * 0.5
    idx = int(np.searchsorted(cum, cutoff, side="left"))
    angle = float(angles_arr[min(idx, angles_arr.size - 1)])

    return {
        "angle": angle,
        "line_count": int(angles_arr.size),
    }


def estimate_rotation(
    mask_bin: np.ndarray,
    min_area_ratio: float = 0.005,
    min_aspect: float = 1.2,
    max_abs_angle: float = 25.0,
    min_abs_angle: float = 0.5,
    strategy: str = "auto",
) -> dict:
    h, w = mask_bin.shape[:2]
    contour = get_largest_contour(mask_bin)
    if contour is None:
        return {
            "angle": 0.0,
            "theta_long": 0.0,
            "center": (w / 2.0, h / 2.0),
            "rect_size": (0.0, 0.0),
            "flag": "fail",
            "reason": "no_contour",
            "applied": False,
            "area_ratio": 0.0,
            "aspect": 0.0,
            "source": strategy,
        }

    area = cv2.contourArea(contour)
    area_ratio = area / max(1.0, float(h * w))
    if area_ratio < min_area_ratio:
        return {
            "angle": 0.0,
            "theta_long": 0.0,
            "center": (w / 2.0, h / 2.0),
            "rect_size": (0.0, 0.0),
            "flag": "fail",
            "reason": "area_too_small",
            "applied": False,
            "area_ratio": area_ratio,
            "aspect": 0.0,
            "source": strategy,
        }

    rect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), angle = rect
    rw = float(rw)
    rh = float(rh)
    if rw <= 0.0 or rh <= 0.0:
        return {
            "angle": 0.0,
            "theta_long": 0.0,
            "center": (cx, cy),
            "rect_size": (rw, rh),
            "flag": "fail",
            "reason": "rect_invalid",
            "applied": False,
            "area_ratio": area_ratio,
            "aspect": 0.0,
            "source": strategy,
        }

    theta_minrect = normalize_long_angle(angle, rw, rh)
    long_side = max(rw, rh)
    short_side = max(1e-6, min(rw, rh))
    aspect = long_side / short_side

    moments = cv2.moments(contour)
    if moments["m00"] != 0.0:
        center_mass = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    else:
        center_mass = (w / 2.0, h / 2.0)
    center_minrect = (cx, cy)

    x, y, bw, bh = cv2.boundingRect(contour)

    def pick_strategy(name: str) -> dict | None:
        if name == "minrect":
            return {"angle": theta_minrect, "source": "minrect", "center": center_minrect}
        if name == "top-edge":
            min_points = max(30, int(bw * 0.2))
            top_edge = estimate_top_edge_angle(
                mask_bin,
                (x, y, bw, bh),
                min_points=min_points,
            )
            if top_edge is None:
                return None
            return {"angle": top_edge["angle"], "source": "top-edge", "center": center_mass}
        if name == "hough":
            hough = estimate_hough_angle(mask_bin, (x, y, bw, bh), max_abs_angle=max_abs_angle)
            if hough is None:
                return None
            return {"angle": hough["angle"], "source": "hough", "center": center_mass}
        if name == "pca":
            pca = estimate_pca_angle(mask_bin)
            if pca is None:
                return None
            return {"angle": pca["angle"], "source": "pca", "center": center_mass}
        return None

    if strategy not in ROTATE_STRATEGIES:
        strategy = "auto"

    picked = None
    if strategy == "auto":
        for name in AUTO_STRATEGY_ORDER:
            picked = pick_strategy(name)
            if picked is not None:
                break
    else:
        picked = pick_strategy(strategy)

    if picked is None:
        return {
            "angle": 0.0,
            "theta_long": 0.0,
            "center": center_mass,
            "rect_size": (rw, rh),
            "flag": "fail",
            "reason": "strategy_failed",
            "applied": False,
            "area_ratio": area_ratio,
            "aspect": aspect,
            "source": strategy,
        }

    source = picked["source"]
    theta_long = picked["angle"]
    angle_to_apply = picked["angle"]
    center = picked["center"]

    flag = "ok"
    if aspect < min_aspect or abs(theta_long) > max_abs_angle:
        flag = "suspicious"

    if abs(theta_long) < min_abs_angle:
        return {
            "angle": 0.0,
            "theta_long": theta_long,
            "center": center,
            "rect_size": (rw, rh),
            "flag": flag,
            "reason": None,
            "applied": False,
            "area_ratio": area_ratio,
            "aspect": aspect,
            "source": source,
        }

    return {
        "angle": angle_to_apply,
        "theta_long": theta_long,
        "center": center,
        "rect_size": (rw, rh),
        "flag": flag,
        "reason": None,
        "applied": True,
        "area_ratio": area_ratio,
        "aspect": aspect,
        "source": source,
    }


def rotate_rgba(
    img: Image.Image,
    alpha_img: Image.Image,
    angle: float,
    center: tuple[float, float],
) -> tuple[Image.Image, Image.Image]:
    img_rgba = img.convert("RGBA")
    img_rgba.putalpha(alpha_img)
    rotated = img_rgba.rotate(
        angle,
        resample=Image.BICUBIC,
        expand=True,
        center=center,
        fillcolor=(255, 255, 255, 0),
    )
    rotated_alpha = rotated.split()[-1]
    return rotated, rotated_alpha


def composite_on_white(img: Image.Image, alpha_img: Image.Image) -> Image.Image:
    canvas = Image.new("RGB", img.size, (255, 255, 255))
    img_rgba = img if img.mode == "RGBA" else img.convert("RGBA")
    canvas.paste(img_rgba, (0, 0), mask=alpha_img)
    return canvas


def process_one(
    img_path: Path,
    out_path: Path,
    mask_path: Path | None,
    rmbg_runner: InspyrenetRunner | None,
    threshold: int,
    close_kernel: int,
    close_iter: int,
    min_angle: float,
    max_angle: float,
    min_aspect: float,
    min_area: float,
    rotate_strategy: str,
    white: bool,
) -> str:
    try:
        img, alpha_img = load_image_and_alpha(img_path, mask_path, rmbg_runner)
    except Exception as exc:
        return f"[SKIP] rmbg failed: {img_path} | {exc}"
    if alpha_img is None:
        return f"[SKIP] no alpha or mask: {img_path}"

    alpha_np = np.array(alpha_img, dtype=np.uint8)
    mask = alpha_np > threshold

    if close_kernel and close_kernel > 0:
        structure = np.ones((close_kernel, close_kernel), dtype=bool)
        mask = ndimage.binary_closing(mask, structure=structure, iterations=close_iter)

    info = estimate_rotation(
        mask,
        min_area_ratio=min_area,
        min_aspect=min_aspect,
        max_abs_angle=max_angle,
        min_abs_angle=min_angle,
        strategy=rotate_strategy,
    )
    if info["applied"]:
        img, alpha_img = rotate_rgba(img, alpha_img, info["angle"], info["center"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if white:
        canvas = composite_on_white(img, alpha_img)
        canvas.save(out_path)
    else:
        img_rgba = img if img.mode == "RGBA" else img.convert("RGBA")
        img_rgba.putalpha(alpha_img)
        img_rgba.save(out_path)

    return (
        f"[OK] {img_path} -> {out_path} | "
        f"angle={info['angle']:.2f} theta={info['theta_long']:.2f} "
        f"flag={info['flag']} area={info['area_ratio']:.4f} aspect={info['aspect']:.2f} "
        f"source={info.get('source', 'n/a')}"
    )


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    mask_dir = Path(args.mask_dir) if args.mask_dir else None

    rmbg_runner = None
    if not args.no_rmbg:
        model_path = Path(args.rmbg_model)
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
        if model_path.exists():
            rmbg_runner = InspyrenetRunner(
                model_path=model_path,
                base_size=tuple(args.rmbg_base_size),
                device=args.rmbg_device,
            )
        else:
            print(f"[WARN] RMBG model not found: {model_path}")

    images = collect_images(input_path, args.recursive)
    if not images:
        print(f"No input images found under: {input_path}")
        return 1

    output_ext = args.output_ext if args.output_ext.startswith(".") else f".{args.output_ext}"

    for img_path in images:
        if input_path.is_dir():
            rel = img_path.relative_to(input_path)
            out_path = output_path / rel
        else:
            out_path = output_path / img_path.name
        out_path = out_path.with_suffix(output_ext)

        mask_path = find_mask_path(img_path, mask_dir, args.mask_suffix)
        msg = process_one(
            img_path=img_path,
            out_path=out_path,
            mask_path=mask_path,
            rmbg_runner=rmbg_runner,
            threshold=args.threshold,
            close_kernel=args.close_kernel,
            close_iter=args.close_iter,
            min_angle=args.min_angle,
            max_angle=args.max_angle,
            min_aspect=args.min_aspect,
            min_area=args.min_area,
            rotate_strategy=args.rotate_strategy,
            white=args.white,
        )
        print(msg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
