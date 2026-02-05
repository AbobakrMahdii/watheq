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
      );
      await _controller!.initialize();
    } catch (e) {
      _error = 'تعذّر تشغيل الكاميرا، تأكد من الإذن';
      debugPrint('Camera error: $e');
    }

    if (mounted) {
      setState(() => _isLoading = false);
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _handleFile(File file) async {
    setState(() => _isProcessing = true);
    final result = DocumentQualityChecker.check(file);
    if (!result.isValid) {
      AppSnackbars.error(context, result.message ?? 'فشل التحقق من الجودة');
      setState(() => _isProcessing = false);
      return;
    }

    if (!mounted) return;
    setState(() => _isProcessing = false);
    Navigator.pop(context, file);
  }

  Future<void> _capture() async {
    if (_controller == null || _isProcessing) return;
    try {
      final image = await _controller!.takePicture();
      await _handleFile(File(image.path));
    } catch (e) {
      AppSnackbars.error(context, 'فشل التقاط الصورة');
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

    return Scaffold(
      appBar: AppBar(title: const Text('التقاط الوثيقة')),
      body: Stack(
        children: [
          CameraPreview(_controller!),
          Center(child: _DocumentFrameOverlay()),
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
              onPressed: _capture,
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

class _DocumentFrameOverlay extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return AspectRatio(
      aspectRatio: 1.6,
      child: Container(
        decoration: BoxDecoration(
          border: Border.all(color: Colors.white, width: 3),
          borderRadius: BorderRadius.circular(12),
        ),
      ),
    );
  }
}
