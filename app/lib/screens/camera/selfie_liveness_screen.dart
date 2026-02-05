import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

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

        if (angleY == null) {
          if (elapsed.inMilliseconds > 2000) {
            _leftDone = true;
            _rightDone = true;
          }
        } else {
          if (!_leftDone && angleY < -12) _leftDone = true;
          if (!_rightDone && angleY > 12) _rightDone = true;
        }

        if (!_blinkDone) {
          if (leftEye != null && rightEye != null) {
            if (leftEye < 0.4 && rightEye < 0.4) _blinkDone = true;
          } else if (elapsed.inMilliseconds > 2000) {
            _blinkDone = true;
          }
        }

        if (_leftDone && _rightDone && _blinkDone) {
          await _finishCapture();
        }
        if (elapsed.inSeconds > 10 && mounted) {
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
    final duration = DateTime.now()
        .difference(_startTime ?? DateTime.now())
        .inMilliseconds;
    final result = SelfieCaptureResult(
      file: File(photo.path),
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
      body: Stack(
        children: [
          CameraPreview(_controller!),
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
                    _StatusRow(label: 'حرّك الرأس يسار', done: _leftDone),
                    _StatusRow(label: 'حرّك الرأس يمين', done: _rightDone),
                    _StatusRow(label: 'ارمش العين', done: _blinkDone),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () async {
          AppSnackbars.info(context, 'اتبع التعليمات لإكمال التحقق');
        },
        child: const Icon(Icons.info_outline),
      ),
    );
  }
}

class _StatusRow extends StatelessWidget {
  const _StatusRow({required this.label, required this.done});

  final String label;
  final bool done;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Icon(
          done ? Icons.check_circle : Icons.radio_button_unchecked,
          color: done ? Colors.green : Colors.grey,
          size: 18,
        ),
        const SizedBox(width: 8),
        Expanded(child: Text(label)),
      ],
    );
  }
}
