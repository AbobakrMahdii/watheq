import 'dart:io';

import 'package:camera/camera.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:pdfx/pdfx.dart';

import '../../features/verification/utils/document_quality_checker.dart';
import '../../ui/widgets/app_snackbars.dart';

class DocumentCaptureScreen extends StatefulWidget {
  const DocumentCaptureScreen({super.key});

  @override
  State<DocumentCaptureScreen> createState() => _DocumentCaptureScreenState();
}

class _DocumentCaptureScreenState extends State<DocumentCaptureScreen> {
  CameraController? _controller;
  bool _isLoading = true;
  bool _isProcessing = false;
  String? _error;

  /// True when live edge analysis detects a card-like rectangle inside the frame.
  bool _cardDetected = false;

  /// Hint shown to the user while they position the card.
  String _hint = 'ضع البطاقة داخل الإطار';

  /// How many consecutive frames the card has been detected.
  int _stableFrames = 0;

  /// Number of stable frames needed before enabling capture.
  static const int _requiredStableFrames = 4;

  /// Debounce: consecutive frames where card is NOT detected.
  int _notDetectedFrames = 0;
  static const int _requiredNotDetectedFrames = 3;

  /// Monotonic counter for unique AnimatedSwitcher keys.
  int _hintSeq = 0;

  bool _isStreaming = false;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      await _controller?.dispose();
      _controller = null;
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        _error = 'لا توجد كاميرا متاحة على هذا الجهاز';
        return;
      }
      final backCamera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      _controller = CameraController(
        backCamera,
        ResolutionPreset.high,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await _controller!.initialize();
      await _startImageStream();
    } catch (e) {
      _error = 'تعذّر تشغيل الكاميرا، تأكد من الإذن';
      debugPrint('Camera error: $e');
    }

    if (mounted) {
      setState(() => _isLoading = false);
    }
  }

  Future<void> _startImageStream() async {
    if (_controller == null || !_controller!.value.isInitialized) return;
    if (_isStreaming) return;
    try {
      await _controller!.startImageStream(_analyzeFrame);
      _isStreaming = true;
    } catch (e) {
      debugPrint('Failed to start image stream: $e');
    }
  }

  Future<void> _stopImageStream() async {
    if (_controller == null || !_isStreaming) return;
    try {
      if (_controller!.value.isStreamingImages) {
        await _controller!.stopImageStream();
      }
    } catch (_) {}
    _isStreaming = false;
  }

  @override
  void dispose() {
    _stopImageStream();
    _controller?.dispose();
    super.dispose();
  }

  // ─── Live frame analysis ───────────────────────────────────────────
  bool _analysing = false;

  void _analyzeFrame(CameraImage frame) {
    if (_analysing || _isProcessing) return;
    _analysing = true;

    final yPlane = frame.planes[0];
    final width = frame.width;
    final height = frame.height;
    final bytes = yPlane.bytes;
    final rowStride = yPlane.bytesPerRow;

    // Define the "frame overlay" region (center 80% width, 1.6 aspect ratio).
    final frameW = (width * 0.80).round();
    final frameH = (frameW / 1.6).round();
    final frameX = ((width - frameW) / 2).round();
    final frameY = ((height - frameH) / 2).round();

    const step = 3; // skip pixels for speed

    // ── 1. Average brightness — reject overexposed / glare / lighter ──
    int brightnessSum = 0;
    int brightnessCount = 0;
    for (var y = frameY; y < frameY + frameH; y += step * 3) {
      for (var x = frameX; x < frameX + frameW; x += step * 3) {
        final idx = y * rowStride + x;
        if (idx >= bytes.length) continue;
        brightnessSum += bytes[idx];
        brightnessCount++;
      }
    }
    final avgBrightness = brightnessCount > 0
        ? brightnessSum ~/ brightnessCount
        : 128;

    // ── 2. Divide frame into 3×3 grid — count zones with edges ──
    // A real ID card has text/photo/patterns spread across the surface.
    // A lighter or lamp only has edges in 1-2 zones.
    const gridCols = 3;
    const gridRows = 3;
    final zoneW = frameW ~/ gridCols;
    final zoneH = frameH ~/ gridRows;
    int zonesWithEdges = 0;

    for (var gr = 0; gr < gridRows; gr++) {
      for (var gc = 0; gc < gridCols; gc++) {
        final zx = frameX + gc * zoneW;
        final zy = frameY + gr * zoneH;
        int zEdges = 0;
        int zTotal = 0;
        for (var y = zy + 2; y < zy + zoneH - 2; y += step) {
          for (var x = zx + 2; x < zx + zoneW - 2; x += step) {
            final idx = y * rowStride + x;
            if (idx + rowStride + 1 >= bytes.length) continue;
            final p = bytes[idx];
            final px = bytes[idx + 1];
            final py = bytes[idx + rowStride];
            final grad = (px - p).abs() + (py - p).abs();
            if (grad > 25) zEdges++;
            zTotal++;
          }
        }
        if (zTotal > 0 && zEdges / zTotal >= 0.02) {
          zonesWithEdges++;
        }
      }
    }

    // ── 3. Border edge density (card edges must be visible) ──
    int borderEdgePixels = 0;
    int borderTotal = 0;
    const borderThickness = 12;

    for (final yStart in [frameY, frameY + frameH - borderThickness]) {
      for (var y = yStart; y < yStart + borderThickness; y += step) {
        for (var x = frameX; x < frameX + frameW; x += step) {
          final idx = y * rowStride + x;
          if (idx + rowStride + 1 >= bytes.length) continue;
          final p = bytes[idx];
          final px = bytes[idx + 1];
          final py = bytes[idx + rowStride];
          final grad = (px - p).abs() + (py - p).abs();
          if (grad > 25) borderEdgePixels++;
          borderTotal++;
        }
      }
    }
    for (final xStart in [frameX, frameX + frameW - borderThickness]) {
      for (var y = frameY; y < frameY + frameH; y += step) {
        for (var x = xStart; x < xStart + borderThickness; x += step) {
          final idx = y * rowStride + x;
          if (idx + rowStride + 1 >= bytes.length) continue;
          final p = bytes[idx];
          final px = bytes[idx + 1];
          final py = bytes[idx + rowStride];
          final grad = (px - p).abs() + (py - p).abs();
          if (grad > 25) borderEdgePixels++;
          borderTotal++;
        }
      }
    }

    final borderDensity = borderTotal > 0
        ? borderEdgePixels / borderTotal
        : 0.0;

    // ── Decision ──
    // Card detected when:
    //   • Not overexposed (avg brightness < 220) — rejects lighters/lamps
    //   • Edges spread across ≥ 5 of 9 zones — rejects small objects
    //   • Border edges visible (≥ 8%) — card edges in frame
    final detected =
        avgBrightness < 220 && zonesWithEdges >= 5 && borderDensity >= 0.08;

    String hint;
    if (avgBrightness >= 220) {
      hint = 'الإضاءة شديدة — أبعد مصدر الضوء';
    } else if (zonesWithEdges < 5) {
      hint = 'ضع البطاقة داخل الإطار';
    } else if (borderDensity < 0.08) {
      hint = 'قرّب البطاقة لتملأ الإطار';
    } else {
      hint = 'ثابت! اضغط لالتقاط الصورة';
    }

    // ── Debounced state transition ──
    if (detected) {
      _stableFrames++;
      _notDetectedFrames = 0;
    } else {
      _notDetectedFrames++;
      // Only drop back to "not detected" after several consecutive misses.
      // This prevents green→white flicker when detection is borderline.
      if (_notDetectedFrames >= _requiredNotDetectedFrames) {
        _stableFrames = 0;
      }
    }

    final readyNow = _stableFrames >= _requiredStableFrames;

    if (mounted && (readyNow != _cardDetected || hint != _hint)) {
      setState(() {
        _cardDetected = readyNow;
        if (hint != _hint) {
          _hint = hint;
          _hintSeq++; // unique key for AnimatedSwitcher
        }
      });
    }

    _analysing = false;
  }

  // ─── Capture & file handling ───────────────────────────────────────

  Future<void> _handleFile(File file) async {
    setState(() => _isProcessing = true);
    final result = DocumentQualityChecker.check(file);
    if (!result.isValid) {
      AppSnackbars.error(context, result.message ?? 'فشل التحقق من الجودة');
      setState(() => _isProcessing = false);
      // Reset detection so the user repositions the card.
      _stableFrames = 0;
      _cardDetected = false;
      await _startImageStream();
      return;
    }

    if (!mounted) return;
    setState(() => _isProcessing = false);
    Navigator.pop(context, file);
  }

  Future<void> _capture() async {
    if (_controller == null || _isProcessing) return;
    try {
      await _stopImageStream();
      final image = await _controller!.takePicture();
      // Pass the full captured image — the card detection already ensures the
      // document fills the frame. Auto-cropping was cutting into the document
      // because the camera sensor aspect ratio differs from the screen preview.
      await _handleFile(File(image.path));
    } catch (e) {
      AppSnackbars.error(context, 'فشل التقاط الصورة');
      // Restart stream so detection resumes.
      await _startImageStream();
    }
  }

  Future<void> _pickFromGallery() async {
    final picker = ImagePicker();
    final file = await picker.pickImage(source: ImageSource.gallery);
    if (file == null) return;
    await _handleFile(File(file.path));
  }

  Future<void> _pickPdf() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf'],
    );
    if (result == null || result.files.isEmpty) return;

    final path = result.files.first.path;
    if (path == null) return;

    try {
      final doc = await PdfDocument.openFile(path);
      final page = await doc.getPage(1);
      final pageImage = await page.render(
        width: page.width,
        height: page.height,
      );
      await page.close();
      await doc.close();

      final tempDir = await getTemporaryDirectory();
      final filePath =
          '${tempDir.path}/document_import_${DateTime.now().millisecondsSinceEpoch}.png';
      final file = File(filePath);
      await file.writeAsBytes(pageImage!.bytes);
      await _handleFile(file);
    } catch (_) {
      AppSnackbars.error(context, 'تعذر قراءة ملف PDF');
    }
  }

  // ─── UI ────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return Scaffold(
        appBar: AppBar(title: const Text('التقاط الوثيقة')),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(_error!, textAlign: TextAlign.center),
                const SizedBox(height: 12),
                ElevatedButton(
                  onPressed: () {
                    setState(() {
                      _isLoading = true;
                      _error = null;
                    });
                    _initCamera();
                  },
                  child: const Text('إعادة المحاولة'),
                ),
              ],
            ),
          ),
        ),
      );
    }

    if (_isLoading || _controller == null) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    final borderColor = _cardDetected ? Colors.green : Colors.white;

    return Scaffold(
      appBar: AppBar(title: const Text('التقاط الوثيقة')),
      body: Stack(
        children: [
          CameraPreview(_controller!),
          // Card detection frame overlay
          Center(
            child: FractionallySizedBox(
              widthFactor: 0.85,
              child: AspectRatio(
                aspectRatio: 1.6,
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 300),
                  decoration: BoxDecoration(
                    border: Border.all(color: borderColor, width: 3),
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
              ),
            ),
          ),
          // Hint text
          Positioned(
            left: 16,
            right: 16,
            bottom: 100,
            child: AnimatedSwitcher(
              duration: const Duration(milliseconds: 200),
              child: Container(
                key: ValueKey(_hintSeq),
                padding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 8,
                ),
                decoration: BoxDecoration(
                  color: (_cardDetected ? Colors.green : Colors.black87)
                      .withOpacity(0.8),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  _hint,
                  textAlign: TextAlign.center,
                  style: const TextStyle(color: Colors.white, fontSize: 14),
                ),
              ),
            ),
          ),
          if (_isProcessing)
            const Positioned.fill(
              child: ColoredBox(
                color: Colors.black54,
                child: Center(child: CircularProgressIndicator()),
              ),
            ),
        ],
      ),
      bottomNavigationBar: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            IconButton(
              onPressed: _pickFromGallery,
              icon: const Icon(Icons.photo_library),
            ),
            FloatingActionButton(
              onPressed: _cardDetected ? _capture : null,
              backgroundColor: _cardDetected ? null : Colors.grey.shade400,
              child: const Icon(Icons.camera_alt),
            ),
            IconButton(
              onPressed: _pickPdf,
              icon: const Icon(Icons.picture_as_pdf),
            ),
          ],
        ),
      ),
    );
  }
}
