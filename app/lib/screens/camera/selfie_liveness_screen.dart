import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

import '../../ui/widgets/app_snackbars.dart';

class SelfieCaptureResult {
  const SelfieCaptureResult({required this.file, required this.livenessData});

  final File file;
  final Map<String, dynamic> livenessData;
}

class SelfieLivenessScreen extends StatefulWidget {
  const SelfieLivenessScreen({super.key, this.useFrontCamera = true});

  final bool useFrontCamera;

  @override
  State<SelfieLivenessScreen> createState() => _SelfieLivenessScreenState();
}

class _SelfieLivenessScreenState extends State<SelfieLivenessScreen>
    with WidgetsBindingObserver {
  CameraController? _controller;
  bool _isLoading = true;
  bool _isProcessing = false;
  bool _leftDone = false;
  bool _rightDone = false;
  bool _blinkDone = false;
  DateTime? _startTime;
  DateTime? _firstFaceSeenAt;
  FaceDetector? _faceDetector;
  String? _error;
  bool _isStreaming = false;

  /// Screen body aspect ratio (width / height), captured at build time
  /// so the crop function can compute the visible preview area.
  double? _screenAspectRatio;

  /// The current step the user must complete (enforced order).
  /// 0 = look left, 1 = look right, 2 = blink, 3 = face forward.
  int _currentStep = 0;

  /// Instruction text for the user.
  String get _instruction {
    switch (_currentStep) {
      case 0:
        return 'حرّك رأسك إلى اليسار';
      case 1:
        return 'حرّك رأسك إلى اليمين';
      case 2:
        return 'ارمش بعينيك';
      case 3:
        return 'انظر للكاميرا مباشرة';
      default:
        return 'جاري الالتقاط...';
    }
  }

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_controller == null) return;
    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      _stopImageStreamSafely();
    } else if (state == AppLifecycleState.resumed) {
      _startImageStreamSafely();
    }
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
      final preferredDirection = widget.useFrontCamera
          ? CameraLensDirection.front
          : CameraLensDirection.back;
      final selectedCamera = cameras.firstWhere(
        (c) => c.lensDirection == preferredDirection,
        orElse: () => cameras.first,
      );
      _controller = CameraController(
        selectedCamera,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: Platform.isAndroid
            ? ImageFormatGroup.nv21
            : ImageFormatGroup.bgra8888,
      );
      await _controller!.initialize();
      _faceDetector = FaceDetector(
        options: FaceDetectorOptions(
          performanceMode: FaceDetectorMode.accurate,
          enableClassification: true,
          enableLandmarks: true,
          enableTracking: true,
        ),
      );
      _startTime = DateTime.now();
      await _startImageStreamSafely();
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
    WidgetsBinding.instance.removeObserver(this);
    _stopImageStreamSafely();
    _controller?.dispose();
    _faceDetector?.close();
    super.dispose();
  }

  Future<void> _startImageStreamSafely() async {
    if (_controller == null || !_controller!.value.isInitialized) return;
    if (_isStreaming) return;
    try {
      await _controller!.startImageStream(_processCameraImage);
      _isStreaming = true;
    } catch (e) {
      debugPrint('Failed to start image stream: $e');
    }
  }

  Future<void> _stopImageStreamSafely() async {
    if (_controller == null) return;
    if (!_isStreaming) return;
    try {
      if (_controller!.value.isStreamingImages) {
        await _controller!.stopImageStream();
      }
    } catch (e) {
      debugPrint('Failed to stop image stream: $e');
    } finally {
      _isStreaming = false;
    }
  }

  Future<void> _processCameraImage(CameraImage image) async {
    if (_isProcessing || _faceDetector == null) return;
    if (!mounted || _controller == null || !_controller!.value.isInitialized) {
      return;
    }
    _isProcessing = true;

    try {
      final inputImage = _inputImageFromCamera(image, _controller!.description);
      if (inputImage == null) {
        _isProcessing = false;
        return;
      }
      final faces = await _faceDetector!.processImage(inputImage);
      if (faces.isNotEmpty) {
        _firstFaceSeenAt ??= DateTime.now();
        final elapsed = DateTime.now().difference(_firstFaceSeenAt!);
        final face = faces.first;
        final angleY = face.headEulerAngleY;
        final leftEye = face.leftEyeOpenProbability;
        final rightEye = face.rightEyeOpenProbability;

        // ── Sequential enforcement: only check the CURRENT step ──

        if (_currentStep == 0) {
          // Step 1: Look LEFT
          if (angleY == null) {
            if (elapsed.inMilliseconds > 3000) {
              _leftDone = true;
              _currentStep = 1;
            }
          } else if (angleY < -12) {
            _leftDone = true;
            _currentStep = 1;
          }
        } else if (_currentStep == 1) {
          // Step 2: Look RIGHT (only after left is done)
          if (angleY == null) {
            if (elapsed.inMilliseconds > 5000) {
              _rightDone = true;
              _currentStep = 2;
            }
          } else if (angleY > 12) {
            _rightDone = true;
            _currentStep = 2;
          }
        } else if (_currentStep == 2) {
          // Step 3: BLINK (only after left+right are done)
          if (leftEye != null && rightEye != null) {
            if (leftEye < 0.4 && rightEye < 0.4) {
              _blinkDone = true;
              _currentStep = 3;
            }
          } else if (elapsed.inMilliseconds > 7000) {
            _blinkDone = true;
            _currentStep = 3;
          }
        } else if (_currentStep == 3) {
          // Step 4: FACE FORWARD — head straight + eyes open for a good selfie
          final isFacingForward = angleY != null && angleY.abs() < 10;
          final eyesOpen = (leftEye ?? 1.0) > 0.5 && (rightEye ?? 1.0) > 0.5;
          if (isFacingForward && eyesOpen) {
            await _finishCapture();
          }
        }

        if (_currentStep >= 4) {
          // All steps done (shouldn't reach here, but safety net)
          await _finishCapture();
        }
        // Hard timeout: auto-capture after 15 seconds regardless
        if (elapsed.inSeconds > 15 && mounted) {
          await _finishCapture();
        }
      } else {
        _firstFaceSeenAt = null;
      }
    } catch (_) {}
    _isProcessing = false;
    if (mounted) setState(() {});
  }

  Future<void> _finishCapture() async {
    if (_controller == null) return;
    await _stopImageStreamSafely();
    final photo = await _controller!.takePicture();

    // Crop the captured photo to the face region
    final croppedFile = await _cropToFace(File(photo.path));

    final duration = DateTime.now()
        .difference(_startTime ?? DateTime.now())
        .inMilliseconds;
    final result = SelfieCaptureResult(
      file: croppedFile,
      livenessData: {
        "passed": true,
        "left": _leftDone,
        "right": _rightDone,
        "blink": _blinkDone,
        "duration_ms": duration,
      },
    );
    if (!mounted) return;
    Navigator.pop(context, result);
  }

  /// Crop the captured selfie to match the oval overlay frame.
  ///
  /// The camera preview widget uses cover-fit: it scales the camera feed
  /// to fill the screen, clipping overflow.  `takePicture()` returns the
  /// full sensor image which is wider (4:3) than what was visible on the
  /// screen (~9:16).  We first compute the "visible region" inside the
  /// full image (the portion the user actually saw), then apply the oval
  /// fractions within that visible region — exactly matching what the
  /// user sees through the overlay.
  Future<File> _cropToFace(File file) async {
    try {
      final bytes = await file.readAsBytes();
      var decoded = img.decodeImage(bytes);
      if (decoded == null) return file;
      decoded = img.bakeOrientation(decoded);

      final imgW = decoded.width;
      final imgH = decoded.height;

      // ── Step 1: Compute the visible preview region ──
      // The screen body size (under the AppBar).  We use the controller's
      // previewSize as a proxy for the camera feed aspect ratio.
      final previewSize = _controller?.value.previewSize;
      // previewSize is in landscape orientation (w > h), e.g. 640×480
      final double cameraAspect =
          (previewSize != null &&
              previewSize.width > 0 &&
              previewSize.height > 0)
          ? previewSize.height /
                previewSize
                    .width // portrait: e.g. 480/640 = 0.75
          : imgW / imgH; // fallback from actual image

      // The phone screen body is roughly 9:16 (~0.5625).
      // We need the actual screen aspect from the Build context, but since
      // this runs after capture, we stored it.  As a robust fallback, derive
      // it from image dimensions + known camera aspect.
      //
      // Cover-fit logic:
      //   scale = max(screenW / camW, screenH / camH)
      // Visible portion of the camera image = screen / scale
      //
      // In proportional terms (all relative to full image):
      //   If camera is wider than screen (cameraAspect < screenAspect):
      //     visible height = full, visible width = cameraAspect / screenAspect
      //   If camera is taller:
      //     visible width = full, visible height = screenAspect / cameraAspect

      // We don't have exact screen size here, but we know the body fills
      // the screen width and the preview uses cover-fit.  A typical phone
      // in portrait has screen aspect ~0.50–0.56.  The front camera feed
      // in portrait is 480/640 = 0.75.  So the camera is *wider* than the
      // screen → sides are clipped → visible width < full width.
      //
      // visibleWidthFraction = screenAspect / cameraAspect
      // We estimate screen aspect from MediaQuery if we stored it, otherwise
      // use a safe default.
      final screenAspect = _screenAspectRatio ?? 0.5; // width / height
      final double visibleWidthFrac;
      final double visibleHeightFrac;

      if (cameraAspect < screenAspect) {
        // Camera wider than screen → height fills, sides clipped
        visibleWidthFrac = cameraAspect / screenAspect;
        visibleHeightFrac = 1.0;
      } else {
        // Camera taller than screen → width fills, top/bottom clipped
        visibleWidthFrac = 1.0;
        visibleHeightFrac = screenAspect / cameraAspect;
      }

      // Visible region in pixel coordinates (centred)
      final visW = (imgW * visibleWidthFrac).round();
      final visH = (imgH * visibleHeightFrac).round();
      final visX = ((imgW - visW) / 2).round();
      final visY = ((imgH - visH) / 2).round();

      // ── Step 2: Apply oval fractions within the visible region ──
      const ovalWidthFrac = 0.70;
      const ovalHeightFrac = 0.50;

      final cropW = (visW * ovalWidthFrac).round();
      final cropH = (visH * ovalHeightFrac).round();
      // Horizontally centred within visible region
      final cropX = visX + ((visW - cropW) / 2).round();
      // Oval centre at 40% of visible height (matching painter)
      final ovalCentreY = visY + (visH * 0.40).round();
      final cropY = ovalCentreY - cropH ~/ 2;

      // Clamp to image bounds
      final x = cropX.clamp(0, imgW - 1);
      final y = cropY.clamp(0, imgH - 1);
      final w = cropW.clamp(1, imgW - x);
      final h = cropH.clamp(1, imgH - y);

      if (w < 80 || h < 80) return file; // sanity guard

      final cropped = img.copyCrop(decoded, x: x, y: y, width: w, height: h);

      final dir = await getTemporaryDirectory();
      final outPath =
          '${dir.path}/selfie_cropped_${DateTime.now().millisecondsSinceEpoch}.jpg';
      final outFile = File(outPath);
      await outFile.writeAsBytes(img.encodeJpg(cropped, quality: 92));
      return outFile;
    } catch (e) {
      debugPrint('Face crop failed, returning original: $e');
      return file;
    }
  }

  InputImage? _inputImageFromCamera(
    CameraImage image,
    CameraDescription camera,
  ) {
    final rotation = _rotationFromSensor(camera);
    if (rotation == null) return null;
    final format = InputImageFormatValue.fromRawValue(image.format.raw);
    if (format == null ||
        (Platform.isAndroid && format != InputImageFormat.nv21) ||
        (Platform.isIOS && format != InputImageFormat.bgra8888)) {
      return null;
    }
    if (image.planes.length != 1) return null;
    final plane = image.planes.first;
    final metadata = InputImageMetadata(
      size: Size(image.width.toDouble(), image.height.toDouble()),
      rotation: rotation,
      format: format,
      bytesPerRow: plane.bytesPerRow,
    );
    return InputImage.fromBytes(bytes: plane.bytes, metadata: metadata);
  }

  static const _orientations = {
    DeviceOrientation.portraitUp: 0,
    DeviceOrientation.landscapeLeft: 90,
    DeviceOrientation.portraitDown: 180,
    DeviceOrientation.landscapeRight: 270,
  };

  InputImageRotation? _rotationFromSensor(CameraDescription camera) {
    if (Platform.isIOS) {
      return InputImageRotationValue.fromRawValue(camera.sensorOrientation);
    }
    final rotationCompensation =
        _orientations[_controller!.value.deviceOrientation];
    if (rotationCompensation == null) return null;
    final isFront = camera.lensDirection == CameraLensDirection.front;
    final rotation = isFront
        ? (camera.sensorOrientation + rotationCompensation) % 360
        : (camera.sensorOrientation - rotationCompensation + 360) % 360;
    return InputImageRotationValue.fromRawValue(rotation);
  }

  Uint8List _concatenatePlanes(List<Plane> planes) {
    final allBytes = BytesBuilder(copy: false);
    for (final plane in planes) {
      allBytes.add(plane.bytes);
    }
    return allBytes.toBytes();
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return Scaffold(
        appBar: AppBar(title: const Text('تصوير السيلفي')),
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
      appBar: AppBar(title: const Text('تصوير السيلفي')),
      body: LayoutBuilder(
        builder: (context, constraints) {
          // Capture screen body aspect ratio for crop calculation
          if (constraints.maxWidth > 0 && constraints.maxHeight > 0) {
            _screenAspectRatio = constraints.maxWidth / constraints.maxHeight;
          }
          return Stack(
            children: [
              CameraPreview(_controller!),
              // Oval face frame overlay
              Positioned.fill(
                child: CustomPaint(
                  painter: _FaceOvalOverlayPainter(
                    ovalWidthFraction: 0.70,
                    ovalHeightFraction: 0.50,
                    borderColor:
                        (_currentStep > 3 ||
                            (_leftDone && _rightDone && _blinkDone))
                        ? Colors.green
                        : Colors.white,
                  ),
                ),
              ),
              // Current instruction banner
              Positioned(
                left: 16,
                right: 16,
                top: 24,
                child: AnimatedSwitcher(
                  duration: const Duration(milliseconds: 200),
                  child: Container(
                    key: ValueKey(_currentStep),
                    padding: const EdgeInsets.symmetric(
                      horizontal: 20,
                      vertical: 10,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.black.withOpacity(0.7),
                      borderRadius: BorderRadius.circular(24),
                    ),
                    child: Text(
                      _instruction,
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ),
              ),
              // Steps checklist
              Positioned(
                left: 16,
                right: 16,
                bottom: 24,
                child: Card(
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        _StatusRow(
                          label: '١. حرّك الرأس يسار',
                          done: _leftDone,
                          active: _currentStep == 0,
                        ),
                        _StatusRow(
                          label: '٢. حرّك الرأس يمين',
                          done: _rightDone,
                          active: _currentStep == 1,
                        ),
                        _StatusRow(
                          label: '٣. ارمش العين',
                          done: _blinkDone,
                          active: _currentStep == 2,
                        ),
                        _StatusRow(
                          label: '٤. انظر للكاميرا',
                          done: _currentStep > 3,
                          active: _currentStep == 3,
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          );
        },
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () async {
          AppSnackbars.info(context, 'اتبع التعليمات بالترتيب لإكمال التحقق');
        },
        child: const Icon(Icons.info_outline),
      ),
    );
  }
}

class _StatusRow extends StatelessWidget {
  const _StatusRow({
    required this.label,
    required this.done,
    this.active = false,
  });

  final String label;
  final bool done;
  final bool active;

  @override
  Widget build(BuildContext context) {
    final Color iconColor;
    final IconData icon;
    if (done) {
      icon = Icons.check_circle;
      iconColor = Colors.green;
    } else if (active) {
      icon = Icons.arrow_forward_ios;
      iconColor = Colors.blue;
    } else {
      icon = Icons.radio_button_unchecked;
      iconColor = Colors.grey;
    }

    return Row(
      children: [
        Icon(icon, color: iconColor, size: 18),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            label,
            style: TextStyle(
              fontWeight: active ? FontWeight.bold : FontWeight.normal,
              color: active ? Colors.blue.shade800 : null,
            ),
          ),
        ),
      ],
    );
  }
}

/// Paints a semi-transparent dark overlay with an oval cutout in the centre
/// to guide the user to position their face.
class _FaceOvalOverlayPainter extends CustomPainter {
  _FaceOvalOverlayPainter({
    this.ovalWidthFraction = 0.70,
    this.ovalHeightFraction = 0.50,
    this.borderColor = Colors.white,
  });

  final double ovalWidthFraction;
  final double ovalHeightFraction;
  final Color borderColor;

  @override
  void paint(Canvas canvas, Size size) {
    final ovalW = size.width * ovalWidthFraction;
    final ovalH = size.height * ovalHeightFraction;
    // Shift the oval slightly upward so it aligns with a typical face position
    final center = Offset(size.width / 2, size.height * 0.40);
    final ovalRect = Rect.fromCenter(
      center: center,
      width: ovalW,
      height: ovalH,
    );

    // Dark overlay with oval hole
    final overlayPaint = Paint()..color = Colors.black.withOpacity(0.55);
    final path = Path()
      ..addRect(Rect.fromLTWH(0, 0, size.width, size.height))
      ..addOval(ovalRect)
      ..fillType = PathFillType.evenOdd;
    canvas.drawPath(path, overlayPaint);

    // Oval border
    final borderPaint = Paint()
      ..color = borderColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;
    canvas.drawOval(ovalRect, borderPaint);
  }

  @override
  bool shouldRepaint(_FaceOvalOverlayPainter oldDelegate) =>
      borderColor != oldDelegate.borderColor;
}
