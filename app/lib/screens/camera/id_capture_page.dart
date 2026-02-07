import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart' as http_parser;
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

class IdCapturePage extends StatefulWidget {
  const IdCapturePage({super.key});

  @override
  State<IdCapturePage> createState() => _IdCapturePageState();
}

class _IdCapturePageState extends State<IdCapturePage> {
  static const double _cardAspect = 856 / 540;
  static const String _baseUrl = 'http://YOUR_BASE_URL'; // TODO: change
  static const String _endpointPath = '/verify';

  CameraController? _controller;
  bool _initializing = true;
  bool _useContainMapping = false;
  Uint8List? _croppedJpeg;
  Rect? _overlayRect;
  Size? _previewSize;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      final back = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final controller = CameraController(
        back,
        ResolutionPreset.high,
        enableAudio: false,
      );
      await controller.initialize();
      setState(() {
        _controller = controller;
        _initializing = false;
      });
    } catch (_) {
      setState(() => _initializing = false);
    }
  }

  Rect _computeOverlayRect(Size size) {
    final double maxW = size.width * 0.80;
    final double maxH = size.height * 0.70;
    double w = maxW;
    double h = w / _cardAspect;
    if (h > maxH) {
      h = maxH;
      w = h * _cardAspect;
    }
    final left = (size.width - w) / 2;
    final top = (size.height - h) / 2;
    return Rect.fromLTWH(left, top, w, h);
  }

  Rect _mapRectScreenToImageCover({
    required Rect overlay,
    required Size widgetSize,
    required Size imageSize,
  }) {
    final scale = math.max(widgetSize.width / imageSize.width, widgetSize.height / imageSize.height);
    final fittedW = imageSize.width * scale;
    final fittedH = imageSize.height * scale;
    final dx = (fittedW - widgetSize.width) / 2;
    final dy = (fittedH - widgetSize.height) / 2;

    final left = (overlay.left + dx) / scale;
    final top = (overlay.top + dy) / scale;
    final right = (overlay.right + dx) / scale;
    final bottom = (overlay.bottom + dy) / scale;
    return Rect.fromLTRB(left, top, right, bottom);
  }

  Rect _mapRectScreenToImageContain({
    required Rect overlay,
    required Size widgetSize,
    required Size imageSize,
  }) {
    final scale = math.min(widgetSize.width / imageSize.width, widgetSize.height / imageSize.height);
    final fittedW = imageSize.width * scale;
    final fittedH = imageSize.height * scale;
    final dx = (widgetSize.width - fittedW) / 2;
    final dy = (widgetSize.height - fittedH) / 2;

    final left = (overlay.left - dx) / scale;
    final top = (overlay.top - dy) / scale;
    final right = (overlay.right - dx) / scale;
    final bottom = (overlay.bottom - dy) / scale;
    return Rect.fromLTRB(left, top, right, bottom);
  }

  Rect _expandRect(Rect r, double pct) {
    final padW = r.width * pct;
    final padH = r.height * pct;
    return Rect.fromLTRB(r.left - padW, r.top - padH, r.right + padW, r.bottom + padH);
  }

  Rect _clampRect(Rect r, int w, int h) {
    final left = r.left.clamp(0.0, w.toDouble());
    final top = r.top.clamp(0.0, h.toDouble());
    final right = r.right.clamp(0.0, w.toDouble());
    final bottom = r.bottom.clamp(0.0, h.toDouble());
    return Rect.fromLTRB(left, top, right, bottom);
  }

  Future<void> _captureCropSend() async {
    final controller = _controller;
    final overlay = _overlayRect;
    final previewSize = _previewSize;
    if (controller == null || overlay == null || previewSize == null) return;
    try {
      final file = await controller.takePicture();
      final bytes = await File(file.path).readAsBytes();
      final decoded = img.decodeImage(bytes);
      if (decoded == null) return;
      img.Image oriented = img.bakeOrientation(decoded);
      if (oriented.height > oriented.width) {
        oriented = img.copyRotate(oriented, angle: 90);
      }
      final imageSize = Size(oriented.width.toDouble(), oriented.height.toDouble());

      final mapped = _useContainMapping
          ? _mapRectScreenToImageContain(
              overlay: overlay,
              widgetSize: previewSize,
              imageSize: imageSize,
            )
          : _mapRectScreenToImageCover(
              overlay: overlay,
              widgetSize: previewSize,
              imageSize: imageSize,
            );
      final padded = _expandRect(mapped, 0.15);
      final clamped = _clampRect(padded, oriented.width, oriented.height);
      final isComplete = (padded.left >= 0 &&
          padded.top >= 0 &&
          padded.right <= oriented.width &&
          padded.bottom <= oriented.height);

      final cropW = clamped.width.round();
      final cropH = clamped.height.round();
      if (cropW <= 0 || cropH <= 0) return;

      final cropped = img.copyCrop(
        oriented,
        x: clamped.left.round(),
        y: clamped.top.round(),
        width: cropW,
        height: cropH,
      );

      final jpeg = Uint8List.fromList(img.encodeJpg(cropped, quality: 92));
      debugPrint(
        '[FLUTTER] croppedBytes=${jpeg.length} croppedW=${cropped.width} croppedH=${cropped.height}',
      );
      setState(() => _croppedJpeg = jpeg);

      final dir = await getApplicationDocumentsDirectory();
      final debugFile = File('${dir.path}/cropped_debug.jpg');
      await debugFile.writeAsBytes(jpeg, flush: true);
      debugPrint('[FLUTTER] saved debug: ${debugFile.path}');

      if (!mounted) return;
      await showDialog(
        context: context,
        builder: (_) => AlertDialog(
          title: const Text('Preview'),
          content: Image.file(debugFile),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Cancel'),
            ),
            ElevatedButton(
              onPressed: isComplete
                  ? () async {
                      Navigator.of(context).pop();
                await _sendImageBytes(jpeg);
                    }
                  : null,
              child: const Text('Send'),
            ),
          ],
        ),
      );
    } catch (_) {}
  }

  Future<void> _sendImageBytes(Uint8List jpgBytes) async {
    final uri = Uri.parse('$_baseUrl$_endpointPath');
    debugPrint('[FLUTTER] croppedBytes length=${jpgBytes.length}');
    final req = http.MultipartRequest('POST', uri);
    req.files.add(
      http.MultipartFile.fromBytes(
        'image',
        jpgBytes,
        filename: 'card.jpg',
        contentType: http_parser.MediaType('image', 'jpeg'),
      ),
    );
    try {
      final res = await req.send();
      final body = await res.stream.bytesToString();
      final snippet = body.length > 300 ? body.substring(0, 300) : body;
      debugPrint('[FLUTTER] status=${res.statusCode} body=$snippet');
    } catch (_) {}
  }

  @override
  Widget build(BuildContext context) {
    if (_initializing) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    if (_controller == null || !_controller!.value.isInitialized) {
      return const Scaffold(body: Center(child: Text('Camera unavailable')));
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('ID Capture'),
        actions: [
          Row(
            children: [
              const Text('Contain'),
              Switch(
                value: _useContainMapping,
                onChanged: (v) => setState(() => _useContainMapping = v),
              ),
            ],
          ),
        ],
      ),
      body: LayoutBuilder(
        builder: (context, constraints) {
          final size = Size(constraints.maxWidth, constraints.maxHeight);
          _overlayRect = _computeOverlayRect(size);
          _previewSize = size;

          return Stack(
            fit: StackFit.expand,
            children: [
              CameraPreview(_controller!),
              CustomPaint(
                painter: _OverlayPainter(_overlayRect!),
              ),
              Positioned(
                bottom: 24,
                left: 24,
                right: 24,
                child: ElevatedButton(
                  onPressed: _captureCropSend,
                  child: const Text('Capture'),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}

class _OverlayPainter extends CustomPainter {
  final Rect rect;
  _OverlayPainter(this.rect);

  @override
  void paint(Canvas canvas, Size size) {
    final overlay = Paint()..color = Colors.black.withOpacity(0.55);
    final clear = Paint()..blendMode = BlendMode.clear;
    final border = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..color = Colors.white;

    canvas.saveLayer(Offset.zero & size, Paint());
    canvas.drawRect(Offset.zero & size, overlay);
    canvas.drawRect(rect, clear);
    canvas.restore();
    canvas.drawRect(rect, border);
  }

  @override
  bool shouldRepaint(covariant _OverlayPainter oldDelegate) {
    return oldDelegate.rect != rect;
  }
}
