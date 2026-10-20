
import uuid
from io import BytesIO
from pathlib import Path

from flask import (
    Flask,
    request,
    render_template,
    send_from_directory,
    send_file,
    jsonify,
    abort,
)
from PIL import (
    Image,
    ImageEnhance,
    ImageFilter,
    UnidentifiedImageError,
)
from werkzeug.utils import secure_filename


# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def readable_size(size):
    """Convert file size into a readable format."""

    if size < 1024:
        return f"{size} bytes"

    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"

    return f"{size / (1024 * 1024):.2f} MB"


def get_safe_upload(filename):
    """Return an existing uploaded image using its basename."""

    safe_name = Path(filename).name
    path = UPLOAD_DIR / safe_name

    if not path.is_file():
        return None

    if path.parent.resolve() != UPLOAD_DIR.resolve():
        return None

    return path


def get_safe_processed(filename):
    """Return an existing processed image using its basename."""

    safe_name = Path(filename).name
    path = PROCESSED_DIR / safe_name

    if not path.is_file():
        return None

    if path.parent.resolve() != PROCESSED_DIR.resolve():
        return None

    return path


def analyze_dominant_colors(image):
    """Find up to six approximate dominant colors."""

    colors = []

    small = image.copy()
    small.thumbnail((250, 250))

    quantized = small.quantize(
        colors=6,
        method=Image.Quantize.MEDIANCUT,
    )

    palette = quantized.getpalette()
    counts = quantized.getcolors() or []

    for count, palette_index in sorted(counts, reverse=True)[:6]:
        start = palette_index * 3
        rgb = tuple(palette[start:start + 3])

        if len(rgb) != 3:
            continue

        colors.append({
            "rgb": f"{rgb[0]}, {rgb[1]}, {rgb[2]}",
            "hex": "#{:02x}{:02x}{:02x}".format(*rgb),
            "count": count,
        })

    return colors


def create_processed_images(image, original_name):
    """Generate grayscale, brightness-default and edge images."""

    grayscale_name = original_name + "_gray.png"
    brightness_name = original_name + "_brightness.png"
    edge_name = original_name + "_edges.png"

    image.convert("L").save(PROCESSED_DIR / grayscale_name)

    ImageEnhance.Brightness(image).enhance(1.0).save(
        PROCESSED_DIR / brightness_name
    )

    image.filter(ImageFilter.FIND_EDGES).save(
        PROCESSED_DIR / edge_name
    )

    return grayscale_name, brightness_name, edge_name


# --------------------------------------------------
# HOME PAGE AND IMAGE UPLOAD
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    original = None
    grayscale = None
    brightness_file = None
    edge_file = None
    info = None
    colors = []

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if not uploaded_file or not uploaded_file.filename:
            error = "Please choose an image first."

        else:
            safe_name = secure_filename(uploaded_file.filename)
            extension = Path(safe_name).suffix.lower()

            if not safe_name or extension not in ALLOWED_EXTENSIONS:
                error = (
                    "Unsupported file. Please upload PNG, JPG, BMP, "
                    "WEBP, or TIFF."
                )

            else:
                original = uuid.uuid4().hex + extension
                upload_path = UPLOAD_DIR / original

                try:
                    # Save the uploaded image.
                    uploaded_file.save(upload_path)

                    # Verify that the file is a valid image.
                    with Image.open(upload_path) as opened:
                        opened.verify()

                    # Read image data after verification.
                    with Image.open(upload_path) as opened:
                        opened.load()
                        original_format = (
                            opened.format or extension.lstrip(".").upper()
                        )
                        original_mode = opened.mode
                        image = opened.convert("RGB")

                    # Generate processing results.
                    (
                        grayscale,
                        brightness_file,
                        edge_file,
                    ) = create_processed_images(image, original)

                    # Calculate image information.
                    width, height = image.size
                    grayscale_image = image.convert("L")

                    brightness = round(
                        sum(grayscale_image.getdata()) / (width * height),
                        1,
                    )

                    info = {
                        "format": original_format,
                        "width": width,
                        "height": height,
                        "total_pixels": width * height,
                        "mode": original_mode,
                        "file_size": readable_size(
                            upload_path.stat().st_size
                        ),
                        "brightness": brightness,
                    }

                    # Detect approximate dominant colors.
                    colors = analyze_dominant_colors(image)

                except (
                    UnidentifiedImageError,
                    OSError,
                    ValueError,
                ):
                    if upload_path.exists():
                        upload_path.unlink(missing_ok=True)

                    original = None
                    error = (
                        "This file is not a valid supported image. "
                        "Please try another image."
                    )

                except Exception:
                    app.logger.exception("Image analysis failed")

                    original = None
                    error = (
                        "An error occurred while analyzing the image. "
                        "Please check the CMD window for details."
                    )

    return render_template(
        "index.html",
        error=error,
        original=original,
        grayscale=grayscale,
        brightness_file=brightness_file,
        edge_file=edge_file,
        info=info,
        colors=colors,
    )


# --------------------------------------------------
# RGB AND HEX PIXEL INSPECTION
# --------------------------------------------------

@app.route("/pixel/<path:filename>")
def pixel_info(filename):
    path = get_safe_upload(filename)

    if path is None:
        return jsonify(error="Image not found"), 404

    try:
        x = int(request.args.get("x", "-1"))
        y = int(request.args.get("y", "-1"))

        with Image.open(path) as image:
            image = image.convert("RGB")

            if (
                x < 0
                or y < 0
                or x >= image.width
                or y >= image.height
            ):
                return jsonify(
                    error="Pixel coordinates are outside the image"
                ), 400

            red, green, blue = image.getpixel((x, y))

        hex_color = "#{:02x}{:02x}{:02x}".format(
            red, green, blue
        )

        return jsonify(
            x=x,
            y=y,
            r=red,
            g=green,
            b=blue,
            hex=hex_color,
        )

    except (ValueError, OSError):
        return jsonify(error="Invalid pixel request"), 400


# --------------------------------------------------
# LIVE BRIGHTNESS ADJUSTMENT
# --------------------------------------------------

@app.route("/brightness/<path:filename>")
def brightness_preview(filename):
    path = get_safe_upload(filename)

    if path is None:
        return "Image not found", 404

    try:
        value = int(request.args.get("value", "100"))
        value = max(0, min(200, value))

        with Image.open(path) as image:
            image = image.convert("RGB")

            adjusted_image = ImageEnhance.Brightness(
                image
            ).enhance(value / 100)

            output = BytesIO()
            adjusted_image.save(output, format="PNG")
            output.seek(0)

            return send_file(
                output,
                mimetype="image/png",
                max_age=0,
            )

    except (ValueError, OSError):
        return "Invalid brightness value", 400


# --------------------------------------------------
# SERVE ORIGINAL UPLOADS
# --------------------------------------------------

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    path = get_safe_upload(filename)

    if path is None:
        abort(404)

    response = send_from_directory(UPLOAD_DIR, path.name)
    response.headers["Cache-Control"] = "no-store"
    return response


# --------------------------------------------------
# SERVE PROCESSED IMAGES
# --------------------------------------------------

@app.route("/processed/<path:filename>")
def processed_file(filename):
    path = get_safe_processed(filename)

    if path is None:
        abort(404)

    response = send_from_directory(PROCESSED_DIR, path.name)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/effects/<path:filename>")
def effect_file(filename):
    path = get_safe_processed(filename)

    if path is None:
        abort(404)

    response = send_from_directory(PROCESSED_DIR, path.name)
    response.headers["Cache-Control"] = "no-store"
    return response


# --------------------------------------------------
# ERROR HANDLERS
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):
    return (
        "File too large. Maximum upload size is 10 MB.",
        413,
    )


@app.errorhandler(404)
def page_not_found(error):
    return "The requested image or page was not found.", 404


# --------------------------------------------------
# START APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5002,
        debug=False,
        use_reloader=False,
    )
