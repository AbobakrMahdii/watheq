import 'dart:io';

import 'package:flutter/material.dart';

import '../../core/constants/app_colors.dart';
import '../../core/constants/app_dimensions.dart';
import '../../features/biometric/models/face_verify_result.dart';
import '../../features/verification/models/ipfs_pin_result.dart';
import '../../features/verification/models/verification_models.dart';
import '../../features/verification/services/verification_orchestrator_service.dart';
import '../../features/verification/services/submit_verification_service.dart';
import '../../features/verification/services/verification_tracker.dart';
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

      // Register with the global tracker so it continues if the user leaves.
      await VerificationTracker.instance.track(record.id);

      var current = record;
      const maxPolls = 120; // stop after ~2 minutes
      var polls = 0;
      while (current.status == VerificationStatus.pending ||
          current.status == VerificationStatus.running) {
        polls++;
        if (polls > maxPolls) {
          throw SubmitVerificationException(
            'انتهت مهلة انتظار التحقق. حاول مرة أخرى لاحقاً.',
          );
        }
        current = await VerificationOrchestratorService.instance.getStatus(
          record.id,
        );
        final steps = await VerificationOrchestratorService.instance.getSteps(
          record.id,
        );
        if (!mounted) return;
        setState(() {
          _steps = steps;
          _currentStage = current.currentStage;
        });
        // Feed the tracker so it doesn't duplicate API calls.
        VerificationTracker.instance.updateFromExternal(current, steps);
        if (current.status == VerificationStatus.pending ||
            current.status == VerificationStatus.running) {
          await Future.delayed(const Duration(seconds: 1));
        }
      }

      if (current.status == VerificationStatus.failed) {
        throw SubmitVerificationException(current.errorMessage ?? 'فشل التحقق');
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

      final ai =
          (data['AI_VERIFICATION'] as Map?)?.cast<String, dynamic>() ?? {};
      _documentDecision = ai['final_decision'] as String?;
      final percent = ai['authenticity_percent'];
      if (percent is num) {
        _documentPercent = percent.toDouble();
      }

      final dataVerification =
          (data['DATA_VERIFICATION'] as Map?)?.cast<String, dynamic>() ?? {};

      final result = SubmitVerificationResult(
        face: face,
        ipfs: ipfs,
        ocr: ocr,
        docId: blockchain['doc_id'] ?? '',
        sha256: blockchain['sha256'] ?? '',
        aiElements: (ai['elements'] as Map?)?.cast<String, dynamic>() ?? {},
        dataVerificationResult: dataVerification,
      );

      if (!mounted) return;
      setState(() {
        _result = result;
        _isLoading = false;
      });
      // Verification finished on-screen — dismiss the tracker (no notification needed).
      await VerificationTracker.instance.dismiss();
    } catch (e) {
      if (!mounted) return;
      final message = e is SubmitVerificationException
          ? e.message
          : e.toString();
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
                            : isActive
                            ? Icons.timelapse
                            : stageIcon(step.stage),
                        size: 18,
                        color: step.status == VerificationStatus.success
                            ? AppColors.success
                            : step.status == VerificationStatus.failed
                            ? AppColors.danger
                            : isActive
                            ? AppColors.primary
                            : AppColors.textSecondary,
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          '${stageArabicLabel(step.stage)} - $statusText',
                          style: TextStyle(
                            fontWeight: isActive
                                ? FontWeight.w800
                                : FontWeight.w500,
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
  final String documentTypeName;
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
        // ---------- Steps summary ----------
        if (steps.isNotEmpty) ...[
          Card(
            child: Padding(
              padding: const EdgeInsets.all(AppDimensions.padLg),
              child: _StepsSummary(steps: steps),
            ),
          ),
          const SizedBox(height: 12),
        ],

        // ---------- Face matching ----------
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
                  'عتبة القبول: ${result.face.acceptThresholdPercent.toStringAsFixed(1)}%',
                  style: const TextStyle(color: AppColors.textSecondary),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 12),

        // ---------- AI Verification (per-element details) ----------
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
                if (documentPercent != null ||
                    (documentDecision ?? '').isNotEmpty)
                  Row(
                    children: [
                      Icon(
                        (documentDecision ?? '').toUpperCase() == 'AUTHENTIC'
                            ? Icons.verified
                            : Icons.info_outline,
                        color:
                            (documentDecision ?? '').toUpperCase() ==
                                'AUTHENTIC'
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
                // Per-element AI results
                if (result.aiElements.isNotEmpty) ...[
                  const SizedBox(height: 10),
                  const Divider(),
                  const Text(
                    'تفاصيل العناصر',
                    style: TextStyle(
                      fontWeight: FontWeight.w700,
                      color: AppColors.textSecondary,
                    ),
                  ),
                  const SizedBox(height: 6),
                  ...result.aiElements.entries.map((entry) {
                    final elementData = entry.value is Map
                        ? entry.value as Map
                        : {};
                    final conf = elementData['confidence'];
                    final status = elementData['status']?.toString() ?? '';
                    final confText = conf is num
                        ? '${(conf * 100).toStringAsFixed(0)}%'
                        : '';
                    final isOk =
                        status.toUpperCase() == 'OK' ||
                        status.toUpperCase() == 'PASS';
                    return Padding(
                      padding: const EdgeInsets.symmetric(vertical: 2),
                      child: Row(
                        children: [
                          Icon(
                            isOk ? Icons.check : Icons.warning_amber,
                            size: 16,
                            color: isOk
                                ? AppColors.success
                                : AppColors.textSecondary,
                          ),
                          const SizedBox(width: 6),
                          Expanded(child: Text(entry.key)),
                          Text(
                            confText,
                            style: const TextStyle(fontWeight: FontWeight.w700),
                          ),
                        ],
                      ),
                    );
                  }),
                ],
              ],
            ),
          ),
        ),
        const SizedBox(height: 12),

        // ---------- Data Verification ----------
        if (result.dataVerificationResult.isNotEmpty)
          Card(
            child: Padding(
              padding: const EdgeInsets.all(AppDimensions.padLg),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  const Text(
                    'مطابقة البيانات',
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.w900),
                  ),
                  const SizedBox(height: 10),
                  // Fraud alert
                  if (result.dataVerificationResult['fraud_suspected'] == true)
                    Container(
                      padding: const EdgeInsets.all(10),
                      margin: const EdgeInsets.only(bottom: 8),
                      decoration: BoxDecoration(
                        color: AppColors.danger.withOpacity(0.1),
                        border: Border.all(color: AppColors.danger),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: const Row(
                        children: [
                          Icon(
                            Icons.warning_amber_rounded,
                            color: AppColors.danger,
                          ),
                          SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              'محاولة احتيال — بيانات الوثيقة لا تطابق السجل المحفوظ',
                              style: TextStyle(
                                color: AppColors.danger,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  // New record info
                  if (result.dataVerificationResult['new_record_created'] ==
                      true)
                    Container(
                      padding: const EdgeInsets.all(10),
                      margin: const EdgeInsets.only(bottom: 8),
                      decoration: BoxDecoration(
                        color: Colors.blue.withOpacity(0.1),
                        border: Border.all(color: Colors.blue),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: const Row(
                        children: [
                          Icon(Icons.person_add, color: Colors.blue),
                          SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              'سجل مواطن جديد — تم حفظ البيانات المستخرجة لأول مرة',
                              style: TextStyle(
                                color: Colors.blue,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  Row(
                    children: [
                      Icon(
                        result.dataVerificationResult['citizen_found'] == true
                            ? Icons.person_search
                            : result.dataVerificationResult['new_record_created'] ==
                                  true
                            ? Icons.person_add
                            : Icons.person_off,
                        color:
                            result.dataVerificationResult['citizen_found'] ==
                                true
                            ? AppColors.success
                            : result.dataVerificationResult['new_record_created'] ==
                                  true
                            ? Colors.blue
                            : AppColors.textSecondary,
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          result.dataVerificationResult['message']
                                  ?.toString() ??
                              '',
                          style: const TextStyle(fontWeight: FontWeight.w600),
                        ),
                      ),
                    ],
                  ),
                  if (result.dataVerificationResult['match_details'] is Map &&
                      (result.dataVerificationResult['match_details'] as Map)
                          .isNotEmpty) ...[
                    const SizedBox(height: 8),
                    ...(result.dataVerificationResult['match_details'] as Map)
                        .entries
                        .map((e) {
                          final detail = e.value is Map ? e.value as Map : {};
                          final matched = detail['match'] == true;
                          return Padding(
                            padding: const EdgeInsets.symmetric(vertical: 2),
                            child: Row(
                              children: [
                                Icon(
                                  matched ? Icons.check : Icons.close,
                                  size: 16,
                                  color: matched
                                      ? AppColors.success
                                      : AppColors.danger,
                                ),
                                const SizedBox(width: 6),
                                Expanded(child: Text(e.key.toString())),
                              ],
                            ),
                          );
                        }),
                  ],
                ],
              ),
            ),
          ),
        if (result.dataVerificationResult.isNotEmpty)
          const SizedBox(height: 12),

        // ---------- Blockchain ----------
        Card(
          child: Padding(
            padding: const EdgeInsets.all(AppDimensions.padLg),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                const Text(
                  'تسجيل البلوكتشين',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.w900),
                ),
                const SizedBox(height: 10),
                Row(
                  children: [
                    const Icon(Icons.shield_outlined, color: AppColors.primary),
                    const SizedBox(width: 8),
                    const Expanded(
                      child: Text(
                        'تم تسجيل بصمة الوثيقة (SHA256) على البلوكتشين وربطها مع CID في IPFS.',
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

        // ---------- OCR ----------
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
                  result.ocr.isEmpty
                      ? 'لا توجد بيانات'
                      : result.ocr['text']?.toString() ?? result.ocr.toString(),
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
                      : stageIcon(step.stage),
                  size: 18,
                  color: step.status == VerificationStatus.success
                      ? AppColors.success
                      : step.status == VerificationStatus.failed
                      ? AppColors.danger
                      : AppColors.textSecondary,
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text('${stageArabicLabel(step.stage)} - $statusText'),
                ),
              ],
            ),
          );
        }),
      ],
    );
  }
}
