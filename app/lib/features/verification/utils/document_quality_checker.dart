import 'dart:io';

import 'package:image/image.dart' as img;

class DocumentQualityResult {
  DocumentQualityResult({
    required this.isBrightnessOk,
    required this.isBlurOk,
    required this.isEdgeOk,
    required this.brightness,
    required this.blurScore,
    required this.edgeScore,
    required this.message,
  });

  final bool isBrightnessOk;
  final bool isBlurOk;
  final bool isEdgeOk;
  final double brightness;
  final double blurScore;
  final double edgeScore;
  final String? message;

  bool get isValid => isBrightnessOk && isBlurOk && isEdgeOk;
}

class DocumentQualityChecker {
  static const double brightnessMin = 30;
  static const double brightnessMax = 235;
  static const double blurMin = 45;
  static const double edgeMin = 0.025;

  static DocumentQualityResult check(File file) {
    final bytes = file.readAsBytesSync();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      return DocumentQualityResult(
        isBrightnessOk: false,
        isBlurOk: false,
        isEdgeOk: false,
        brightness: 0,
        blurScore: 0,
        edgeScore: 0,
        message: 'تعذر قراءة الصورة',
      );
    }

    final resized = img.copyResize(decoded, width: 256);
    final gray = img.grayscale(resized);

    final brightness = _averageBrightness(gray);
    final blur = _laplacianVariance(gray);
    final edge = _edgeDensity(gray);

    final brightnessOk =
        brightness >= brightnessMin && brightness <= brightnessMax;
    final blurOk = blur >= blurMin;
    final edgeOk = edge >= edgeMin;

    String? message;
    if (!brightnessOk) {
      message = 'الإضاءة غير مناسبة';
    } else if (!blurOk) {
      message = 'نظّف الكاميرا';
    } else if (!edgeOk) {
      message = 'حرّك البطاقة داخل الإطار';
    }

    return DocumentQualityResult(
      isBrightnessOk: brightnessOk,
      isBlurOk: blurOk,
      isEdgeOk: edgeOk,
      brightness: brightness,
      blurScore: blur,
      edgeScore: edge,
      message: message,
    );
  }

  static double _averageBrightness(img.Image image) {
    var total = 0.0;
    final count = image.width * image.height;
    for (var y = 0; y < image.height; y++) {
      for (var x = 0; x < image.width; x++) {
        final pixel = image.getPixel(x, y);
        total += img.getLuminance(pixel);
      }
    }
    return total / count;
  }

  static double _laplacianVariance(img.Image image) {
    final kernel = [
      [0, 1, 0],
      [1, -4, 1],
      [0, 1, 0],
    ];
    final values = <double>[];
    for (var y = 1; y < image.height - 1; y++) {
      for (var x = 1; x < image.width - 1; x++) {
        var sum = 0.0;
        for (var ky = -1; ky <= 1; ky++) {
          for (var kx = -1; kx <= 1; kx++) {
            final pixel = image.getPixel(x + kx, y + ky);
            final value = img.getLuminance(pixel);
            sum += kernel[ky + 1][kx + 1] * value;
          }
        }
        values.add(sum);
      }
    }

    if (values.isEmpty) return 0;
    final mean = values.reduce((a, b) => a + b) / values.length;
    final variance =
        values.map((v) => (v - mean) * (v - mean)).reduce((a, b) => a + b) /
        values.length;
    return variance;
  }

  static double _edgeDensity(img.Image image) {
    final width = image.width;
    final height = image.height;
    var edges = 0;
    var total = 0;
    for (var y = 1; y < height - 1; y++) {
      for (var x = 1; x < width - 1; x++) {
        final p = img.getLuminance(image.getPixel(x, y));
        final px = img.getLuminance(image.getPixel(x + 1, y));
        final py = img.getLuminance(image.getPixel(x, y + 1));
        final dx = (px - p).abs();
        final dy = (py - p).abs();
        if (dx + dy > 30) edges++;
        total++;
      }
    }
    if (total == 0) return 0;
    return edges / total;
  }
}
