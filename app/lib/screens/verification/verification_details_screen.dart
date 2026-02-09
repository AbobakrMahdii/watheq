import 'package:flutter/material.dart';

import '../../core/constants/app_colors.dart';
import '../../core/constants/app_dimensions.dart';
import '../../features/auth/models/user_profile.dart';
import '../../features/auth/services/auth_service.dart';
import '../../features/verification/models/verification_models.dart';
import '../../features/verification/services/verification_orchestrator_service.dart';
import '../../features/verification/services/verification_tracker.dart';
import '../../ui/widgets/app_snackbars.dart';

/// Read-only screen showing full verification details.
///
/// Accepts a [verificationId] via constructor or route arguments.
/// Fetches the verification record + steps and renders them.
class VerificationDetailsScreen extends StatefulWidget {
  const VerificationDetailsScreen({super.key, required this.verificationId});

  final int verificationId;

  @override
  State<VerificationDetailsScreen> createState() =>
      _VerificationDetailsScreenState();
}

class _VerificationDetailsScreenState extends State<VerificationDetailsScreen> {
  bool _loading = true;
  VerificationRecord? _record;
  List<VerificationStep> _steps = [];
  String? _error;
  UserProfile? _profile;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final record = await VerificationOrchestratorService.instance.getStatus(
        widget.verificationId,
      );
      final steps = await VerificationOrchestratorService.instance.getSteps(
        widget.verificationId,
      );
      final profile = await AuthService.instance.fetchProfile();
      if (!mounted) return;
      setState(() {
        _record = record;
        _steps = steps;
        _profile = profile;
        _loading = false;
      });

      // If the user came here from the FAB or notification, dismiss tracker.
      final tracker = VerificationTracker.instance;
      if (tracker.activeId == widget.verificationId &&
          tracker.finalStatus != null) {
        tracker.dismiss();
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _loading = false;
      });
      AppSnackbars.error(context, _error!);
    }
  }

  bool _isAdmin() {
    final role = _profile?.role ?? '';
    return role == 'super_admin' || role == 'admin';
  }

  // ─── helpers ──────────────────────────────────────────────────────

  String _statusLabel(VerificationStatus s) {
    switch (s) {
      case VerificationStatus.pending:
        return 'في الانتظار';
      case VerificationStatus.running:
        return 'قيد التنفيذ';
      case VerificationStatus.success:
        return 'ناجح';
      case VerificationStatus.failed:
        return 'فاشل';
    }
  }

  Color _statusColor(VerificationStatus s) {
    switch (s) {
      case VerificationStatus.success:
        return AppColors.success;
      case VerificationStatus.failed:
        return AppColors.danger;
      default:
        return AppColors.warning;
    }
  }

  IconData _statusIcon(VerificationStatus s) {
    switch (s) {
      case VerificationStatus.success:
        return Icons.check_circle;
      case VerificationStatus.failed:
        return Icons.error;
      default:
        return Icons.timelapse;
    }
  }

  String _stepStatusText(VerificationStatus s) {
    switch (s) {
      case VerificationStatus.success:
        return 'تم';
      case VerificationStatus.failed:
        return 'فشل';
      case VerificationStatus.running:
        return 'قيد التنفيذ';
      case VerificationStatus.pending:
        return 'في الانتظار';
    }
  }

  // ─── build ────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('تفاصيل التحقق #${widget.verificationId}'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _load,
            tooltip: 'تحديث',
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
          ? Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(_error!, textAlign: TextAlign.center),
                  const SizedBox(height: 12),
                  ElevatedButton(
                    onPressed: _load,
                    child: const Text('إعادة المحاولة'),
                  ),
                ],
              ),
            )
          : _buildContent(),
    );
  }

  Widget _buildContent() {
    final record = _record!;
    final data = record.resultData ?? {};

    return ListView(
      padding: const EdgeInsets.all(AppDimensions.padLg),
      children: [
        // ── Status banner ──
        Card(
          color: _statusColor(record.status).withOpacity(0.08),
          child: Padding(
            padding: const EdgeInsets.all(AppDimensions.padLg),
            child: Row(
              children: [
                Icon(
                  _statusIcon(record.status),
                  color: _statusColor(record.status),
                  size: 36,
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        _statusLabel(record.status),
                        style: TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.w900,
                          color: _statusColor(record.status),
                        ),
                      ),
                      if (record.errorMessage != null &&
                          record.errorMessage!.isNotEmpty)
                        Padding(
                          padding: const EdgeInsets.only(top: 4),
                          child: Text(
                            record.errorMessage!,
                            style: const TextStyle(color: AppColors.danger),
                          ),
                        ),
                      if (record.createdAt != null)
                        Padding(
                          padding: const EdgeInsets.only(top: 4),
                          child: Text(
                            'تاريخ الإنشاء: ${_formatDate(record.createdAt!)}',
                            style: const TextStyle(
                              color: AppColors.textSecondary,
                              fontSize: 12,
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 16),

        // ── Steps ──
        if (_steps.isNotEmpty)
          Card(
            child: Padding(
              padding: const EdgeInsets.all(AppDimensions.padLg),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  const Text(
                    'مراحل التحقق',
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.w900),
                  ),
                  const SizedBox(height: 10),
                  ..._steps.map(_buildStepRow),
                ],
              ),
            ),
          ),
        const SizedBox(height: 12),

        // ── Face matching ──
        if (data.containsKey('FACE_MATCHING') || data.containsKey('BIOMETRIC'))
          _buildFaceCard(data),

        // ── AI Verification ──
        if (data.containsKey('AI_VERIFICATION')) _buildAiCard(data),

        // ── Data Verification ──
        if (data.containsKey('DATA_VERIFICATION'))
          _buildDataVerificationCard(data),

        // ── Blockchain ──
        if (_isAdmin() && data.containsKey('BLOCKCHAIN'))
          _buildBlockchainCard(data),

        // ── OCR ──
        if (data.containsKey('OCR')) _buildOcrCard(data),

        const SizedBox(height: 24),
      ],
    );
  }

  Widget _buildStepRow(VerificationStep step) {
    final isActive = _record?.currentStage == step.stage;
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
            size: 20,
            color: step.status == VerificationStatus.success
                ? AppColors.success
                : step.status == VerificationStatus.failed
                ? AppColors.danger
                : isActive
                ? AppColors.primary
                : AppColors.textSecondary,
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              stageArabicLabel(step.stage),
              style: TextStyle(
                fontWeight: isActive ? FontWeight.w800 : FontWeight.w500,
              ),
            ),
          ),
          Text(
            _stepStatusText(step.status),
            style: TextStyle(
              color: step.status == VerificationStatus.success
                  ? AppColors.success
                  : step.status == VerificationStatus.failed
                  ? AppColors.danger
                  : AppColors.textSecondary,
              fontWeight: FontWeight.w600,
              fontSize: 12,
            ),
          ),
          if (step.errorMessage != null && step.errorMessage!.isNotEmpty) ...[
            const SizedBox(width: 4),
            Tooltip(
              message: step.errorMessage!,
              child: const Icon(
                Icons.info_outline,
                size: 14,
                color: AppColors.danger,
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildFaceCard(Map<String, dynamic> data) {
    final faceData =
        (data['FACE_MATCHING'] as Map?)?.cast<String, dynamic>() ??
        (data['BIOMETRIC'] as Map?)?.cast<String, dynamic>() ??
        {};
    final similarity =
        ((faceData['similarity_percent'] ?? faceData['score'] ?? 0) as num)
            .toDouble();
    final isMatch = faceData['match'] == true || faceData['accepted'] == true;

    return Column(
      children: [
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
                      '${similarity.toStringAsFixed(1)}%',
                      style: const TextStyle(
                        fontWeight: FontWeight.w900,
                        fontSize: 16,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 12),
      ],
    );
  }

  Widget _buildAiCard(Map<String, dynamic> data) {
    final ai = (data['AI_VERIFICATION'] as Map?)?.cast<String, dynamic>() ?? {};
    final decision = ai['final_decision'] as String?;
    final percent = ai['authenticity_percent'];
    final percentValue = percent is num ? percent.toDouble() : null;
    final elements = (ai['elements'] as Map?)?.cast<String, dynamic>() ?? {};

    return Column(
      children: [
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
                if (percentValue != null || (decision ?? '').isNotEmpty)
                  Row(
                    children: [
                      Icon(
                        (decision ?? '').toUpperCase() == 'AUTHENTIC'
                            ? Icons.verified
                            : Icons.info_outline,
                        color: (decision ?? '').toUpperCase() == 'AUTHENTIC'
                            ? AppColors.success
                            : AppColors.textSecondary,
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          [
                            if ((decision ?? '').isNotEmpty)
                              'القرار: $decision',
                            if (percentValue != null)
                              'النسبة: ${percentValue.toStringAsFixed(1)}%',
                          ].join(' • '),
                        ),
                      ),
                    ],
                  ),
                if (elements.isNotEmpty) ...[
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
                  ...elements.entries.map((entry) {
                    final eData = entry.value is Map ? entry.value as Map : {};
                    final conf = eData['confidence'];
                    final status = eData['status']?.toString() ?? '';
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
      ],
    );
  }

  Widget _buildDataVerificationCard(Map<String, dynamic> data) {
    final dv =
        (data['DATA_VERIFICATION'] as Map?)?.cast<String, dynamic>() ?? {};
    if (dv.isEmpty) return const SizedBox.shrink();

    return Column(
      children: [
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
                if (dv['fraud_suspected'] == true)
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
                if (dv['new_record_created'] == true)
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
                      dv['citizen_found'] == true
                          ? Icons.person_search
                          : dv['new_record_created'] == true
                          ? Icons.person_add
                          : Icons.person_off,
                      color: dv['citizen_found'] == true
                          ? AppColors.success
                          : dv['new_record_created'] == true
                          ? Colors.blue
                          : AppColors.textSecondary,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        dv['message']?.toString() ?? '',
                        style: const TextStyle(fontWeight: FontWeight.w600),
                      ),
                    ),
                  ],
                ),
                if (dv['match_details'] is Map &&
                    (dv['match_details'] as Map).isNotEmpty) ...[
                  const SizedBox(height: 8),
                  ...(dv['match_details'] as Map).entries.map((e) {
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
        const SizedBox(height: 12),
      ],
    );
  }

  Widget _buildBlockchainCard(Map<String, dynamic> data) {
    final bc = (data['BLOCKCHAIN'] as Map?)?.cast<String, dynamic>() ?? {};
    return Column(
      children: [
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
                _kv('DocId', bc['doc_id']?.toString() ?? '—'),
                _kv('CID', bc['cid']?.toString() ?? '—'),
                _kv('SHA256', bc['sha256']?.toString() ?? '—'),
              ],
            ),
          ),
        ),
        const SizedBox(height: 12),
      ],
    );
  }

  Widget _buildOcrCard(Map<String, dynamic> data) {
    final ocr = (data['OCR'] as Map?)?.cast<String, dynamic>() ?? {};
    return Card(
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
              ocr.isEmpty
                  ? 'لا توجد بيانات'
                  : ocr['text']?.toString() ?? ocr.toString(),
              style: const TextStyle(color: AppColors.textSecondary),
            ),
          ],
        ),
      ),
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
          Expanded(child: Text(v, style: const TextStyle(fontSize: 12))),
        ],
      ),
    );
  }

  String _formatDate(DateTime dt) {
    return '${dt.year}-${dt.month.toString().padLeft(2, '0')}-${dt.day.toString().padLeft(2, '0')}'
        '  ${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';
  }
}
