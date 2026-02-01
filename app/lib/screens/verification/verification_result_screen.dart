import 'dart:io';

import 'package:flutter/material.dart';

import '../../core/constants/app_colors.dart';
import '../../core/constants/app_dimensions.dart';
import '../../features/biometric/models/face_verify_result.dart';
import '../../features/verification/models/ipfs_pin_result.dart';
import '../../features/verification/models/verification_models.dart';
import '../../features/verification/services/verification_orchestrator_service.dart';
import '../../features/verification/services/submit_verification_service.dart';
import '../../ui/widgets/app_snackbars.dart';

class VerificationResultScreen extends StatefulWidget {
  const VerificationResultScreen({
    super.key,
    required this.documentImageFront,
    this.documentImageBack,
    required this.personImage,
    required this.documentTypeId,
    required this.documentTypeName,
    this.livenessData,
  });

  final File documentImageFront;
  final File? documentImageBack; // Optional
  final File personImage;
  final int documentTypeId; // Now an int
  final String documentTypeName; // Display name
  final Map<String, dynamic>? livenessData;

  @override
  State<VerificationResultScreen> createState() =>
      _VerificationResultScreenState();
}

class _VerificationResultScreenState extends State<VerificationResultScreen> {
  bool _isLoading = true;
  SubmitVerificationResult? _result;
  String? _documentDecision;
  double? _documentPercent;
  String? _error;
  List<VerificationStep> _steps = [];
  VerificationStage? _currentStage;

  @override
  void initState() {
    super.initState();
    _run();
  }

  Future<void> _run() async {
    setState(() {
      _isLoading = true;
      _error = null;
      _result = null;
      _steps = [];
      _currentStage = null;
    });

    try {
      // Start sequential verification pipeline on the backend (no API calls before button press).
      final record = await VerificationOrchestratorService.instance.start(
        documentImageFront: widget.documentImageFront,
        documentImageBack: widget.documentImageBack,
        personImage: widget.personImage,
        documentTypeId: widget.documentTypeId,
        livenessData: widget.livenessData,
      );

      var current = record;
      while (current.status == VerificationStatus.pending ||
          current.status == VerificationStatus.running) {
        current =
            await VerificationOrchestratorService.instance.getStatus(record.id);
        final steps =
            await VerificationOrchestratorService.instance.getSteps(record.id);
        if (!mounted) return;
        setState(() {
          _steps = steps;
          _currentStage = current.currentStage;
        });
        if (current.status == VerificationStatus.running) {
          await Future.delayed(const Duration(seconds: 1));
        }
      }

      if (current.status == VerificationStatus.failed) {
        throw SubmitVerificationException(
          current.errorMessage ?? 'فشل التحقق',
        );
      }

      final data = current.resultData ?? {};
      final facePayload =
          (data['FACE_MATCHING'] as Map?)?.cast<String, dynamic>() ??
              (data['BIOMETRIC'] as Map?)?.cast<String, dynamic>() ??
              {};
      final face = FaceVerifyResult.fromJson(facePayload);
      final ocr = (data['OCR'] as Map?)?.cast<String, dynamic>() ?? {};
      final blockchain =
          (data['BLOCKCHAIN'] as Map?)?.cast<String, dynamic>() ?? {};
      final ipfs = IpfsPinResult.fromJson({
        'cid': blockchain['cid'] ?? '',
        'filename': blockchain['filename'] ?? '',
      });

      final ml = (data['ML'] as Map?)?.cast<String, dynamic>() ?? {};
      _documentDecision = ml['final_decision'] as String?;
      final percent = ml['authenticity_percent'];
      if (percent is num) {
        _documentPercent = percent.toDouble();
      }

      final result = SubmitVerificationResult(
        face: face,
        ipfs: ipfs,
        ocr: ocr,
        docId: blockchain['doc_id'] ?? '',
        sha256: blockchain['sha256'] ?? '',
        ledgerRecorded: blockchain['ledger_recorded'] == true,
      );

      if (!mounted) return;
      setState(() {
        _result = result;
        _isLoading = false;
      });
    } catch (e) {
      if (!mounted) return;
      final message = e is SubmitVerificationException ? e.message : e.toString();
      setState(() {
        _error = message;
        _isLoading = false;
      });
      AppSnackbars.error(context, _error!);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('نتائج التحقق')),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 640),
          child: Padding(
            padding: const EdgeInsets.all(AppDimensions.padLg),
            child: _isLoading
                ? _ProgressView(steps: _steps, currentStage: _currentStage)
                : _error != null
                    ? _ErrorView(message: _error!, onRetry: _run, steps: _steps)
                    : _ResultView(
                        result: _result!,
                        documentTypeName: widget.documentTypeName,
                        documentDecision: _documentDecision,
                        documentPercent: _documentPercent,
                        steps: _steps,
                      ),
          ),
        ),
      ),
    );
  }
}

class _ProgressView extends StatelessWidget {
  const _ProgressView({required this.steps, required this.currentStage});

  final List<VerificationStep> steps;
  final VerificationStage? currentStage;

  @override
  Widget build(BuildContext context) {
    String labelForStage(VerificationStage stage) {
      switch (stage) {
        case VerificationStage.documentImageQuality:
          return 'Document Quality';
        case VerificationStage.documentCropping:
          return 'Document Cropping';
        case VerificationStage.documentFaceExtraction:
          return 'Document Face';
        case VerificationStage.selfieLiveness:
          return 'Selfie Liveness';
        case VerificationStage.faceMatching:
          return 'Face Matching';
        case VerificationStage.biometric:
          return 'Biometric';
        case VerificationStage.ml:
          return 'ML';
        case VerificationStage.ocr:
          return 'OCR';
        case VerificationStage.blockchain:
          return 'Blockchain';
      }
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(AppDimensions.padLg),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const CircularProgressIndicator(),
            const SizedBox(height: 12),
            const Text(
              'جاري تنفيذ مراحل التحقق...',
              style: TextStyle(fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 12),
            if (steps.isNotEmpty)
              ...steps.map((step) {
                final isActive = currentStage == step.stage;
                final statusText = step.status.name.toUpperCase();
                return Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Row(
                    children: [
                      Icon(
                        step.status == VerificationStatus.success
                            ? Icons.check_circle
                            : step.status == VerificationStatus.failed
                                ? Icons.error
                                : Icons.timelapse,
                        size: 18,
                        color: step.status == VerificationStatus.success
                            ? AppColors.success
                            : step.status == VerificationStatus.failed
                                ? AppColors.danger
                                : AppColors.textSecondary,
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          '${labelForStage(step.stage)} - $statusText',
                          style: TextStyle(
                            fontWeight: isActive ? FontWeight.w800 : FontWeight.w500,
                          ),
                        ),
                      ),
                    ],
                  ),
                );
              }),
          ],
        ),
      ),
    );
  }
}

class _ErrorView extends StatelessWidget {
  const _ErrorView({
    required this.message,
    required this.onRetry,
    required this.steps,
  });

  final String message;
  final VoidCallback onRetry;
  final List<VerificationStep> steps;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(AppDimensions.padLg),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'فشل التحقق',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.w900),
            ),
            const SizedBox(height: 10),
            Text(message, style: const TextStyle(color: AppColors.danger)),
            if (steps.isNotEmpty) ...[
              const SizedBox(height: 16),
              _StepsSummary(steps: steps),
            ],
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: onRetry,
              icon: const Icon(Icons.refresh),
              label: const Text('إعادة المحاولة'),
            ),
          ],
        ),
      ),
    );
  }
}

class _ResultView extends StatelessWidget {
  const _ResultView({
    required this.result,
    required this.documentTypeName,
    required this.documentDecision,
    required this.documentPercent,
    required this.steps,
  });

  final SubmitVerificationResult result;
  final String documentTypeName; // Display name
  final String? documentDecision;
  final double? documentPercent;
  final List<VerificationStep> steps;

  @override
  Widget build(BuildContext context) {
    final similarity = result.face.similarityPercent;
    final similarityText = '${similarity.toStringAsFixed(1)}%';
    final isMatch = result.face.match;

    return ListView(
      children: [
        if (steps.isNotEmpty) ...[
          Card(
            child: Padding(
              padding: const EdgeInsets.all(AppDimensions.padLg),
              child: _StepsSummary(steps: steps),
            ),
          ),
          const SizedBox(height: 12),
        ],
        Card(
          child: Padding(
            padding: const EdgeInsets.all(AppDimensions.padLg),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                const Text(
                  'نتيجة التطابق',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.w900),
                ),
                const SizedBox(height: 10),
                Row(
                  children: [
                    Icon(
                      isMatch ? Icons.verified : Icons.error_outline,
                      color: isMatch ? AppColors.success : AppColors.danger,
                      size: 24,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      isMatch ? 'مطابق' : 'غير مطابق',
                      style: TextStyle(
                        fontWeight: FontWeight.w800,
                        color: isMatch ? AppColors.success : AppColors.danger,
                      ),
                    ),
                    const Spacer(),
                    Text(
                      similarityText,
                      style: const TextStyle(
                        fontWeight: FontWeight.w900,
                        fontSize: 16,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Text(
                  'النموذج: ${result.face.model} • المعيار: ${result.face.distanceMetric}',
                  style: const TextStyle(color: AppColors.textSecondary),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 12),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(AppDimensions.padLg),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                const Text(
                  'أصالة الوثيقة',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.w900),
                ),
                const SizedBox(height: 10),
                if (documentPercent != null || (documentDecision ?? '').isNotEmpty)
                  Row(
                    children: [
                      Icon(
                        (documentDecision ?? '').toUpperCase() == 'AUTHENTIC'
                            ? Icons.verified
                            : Icons.info_outline,
                        color: (documentDecision ?? '').toUpperCase() == 'AUTHENTIC'
                            ? AppColors.success
                            : AppColors.textSecondary,
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          [
                            if ((documentDecision ?? '').isNotEmpty)
                              'القرار: $documentDecision',
                            if (documentPercent != null)
                              'النسبة: ${documentPercent!.toStringAsFixed(1)}%',
                          ].join(' • '),
                        ),
                      ),
                    ],
                  ),
                if (documentPercent != null || (documentDecision ?? '').isNotEmpty)
                  const SizedBox(height: 10),
                Row(
                  children: [
                    const Icon(Icons.shield_outlined, color: AppColors.primary),
                    const SizedBox(width: 8),
                    const Expanded(
                      child: Text(
                        'تم تسجيل بصمة الوثيقة (SHA256) على السجل (Blockchain) وربطها مع CID في IPFS.',
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                _kv('نوع الهوية', documentTypeName),
                _kv('DocId', result.docId),
                _kv('CID', result.ipfs.cid),
                _kv('SHA256', result.sha256),
              ],
            ),
          ),
        ),
        const SizedBox(height: 12),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(AppDimensions.padLg),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                const Text(
                  'بيانات OCR',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.w900),
                ),
                const SizedBox(height: 10),
                Text(
                  result.ocr.isEmpty ? 'لا توجد بيانات' : result.ocr.toString(),
                  style: const TextStyle(color: AppColors.textSecondary),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _kv(String k, String v) {
    return Padding(
      padding: const EdgeInsets.only(top: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 80,
            child: Text(
              k,
              style: const TextStyle(
                fontWeight: FontWeight.w800,
                color: AppColors.textSecondary,
              ),
            ),
          ),
          const SizedBox(width: 8),
          Expanded(child: Text(v)),
        ],
      ),
    );
  }
}

class _StepsSummary extends StatelessWidget {
  const _StepsSummary({required this.steps});

  final List<VerificationStep> steps;

  String _label(VerificationStage stage) {
    switch (stage) {
      case VerificationStage.documentImageQuality:
        return 'Document Quality';
      case VerificationStage.documentCropping:
        return 'Document Cropping';
      case VerificationStage.documentFaceExtraction:
        return 'Document Face';
      case VerificationStage.selfieLiveness:
        return 'Selfie Liveness';
      case VerificationStage.faceMatching:
        return 'Face Matching';
      case VerificationStage.biometric:
        return 'Biometric';
      case VerificationStage.ml:
        return 'ML';
      case VerificationStage.ocr:
        return 'OCR';
      case VerificationStage.blockchain:
        return 'Blockchain';
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const Text(
          'مراحل التحقق',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.w900),
        ),
        const SizedBox(height: 10),
        ...steps.map((step) {
          final statusText = step.status == VerificationStatus.success
              ? 'تم'
              : step.status == VerificationStatus.failed
                  ? 'فشل'
                  : step.status == VerificationStatus.running
                      ? 'قيد التنفيذ'
                      : 'في الانتظار';
          return Padding(
            padding: const EdgeInsets.symmetric(vertical: 4),
            child: Row(
              children: [
                Icon(
                  step.status == VerificationStatus.success
                      ? Icons.check_circle
                      : step.status == VerificationStatus.failed
                          ? Icons.error
                          : Icons.timelapse,
                  size: 18,
                  color: step.status == VerificationStatus.success
                      ? AppColors.success
                      : step.status == VerificationStatus.failed
                          ? AppColors.danger
                          : AppColors.textSecondary,
                ),
                const SizedBox(width: 8),
                Expanded(child: Text('${_label(step.stage)} - $statusText')),
              ],
            ),
          );
        }),
      ],
    );
  }
}
