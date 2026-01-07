import 'dart:io';

import 'package:flutter/material.dart';

import '../../core/constants/app_colors.dart';
import '../../core/constants/app_dimensions.dart';
import '../../features/verification/services/document_verify_service.dart';
import '../../features/verification/services/submit_verification_service.dart';
import '../../ui/widgets/app_snackbars.dart';

class VerificationResultScreen extends StatefulWidget {
  const VerificationResultScreen({
    super.key,
    required this.documentImage,
    required this.personImage,
    required this.idType,
  });

  final File documentImage;
  final File personImage;
  final String idType;

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
    });

    try {
      // Run document authenticity verification (best-effort, does not block submit).
      try {
        final docRes = await DocumentVerifyService.instance.verify(
          documentImage: widget.documentImage,
        );
        if (mounted) {
          setState(() {
            _documentDecision = docRes.finalDecision;
            _documentPercent = docRes.authenticityPercent;
          });
        }
      } catch (_) {
        // Ignore to keep flow working even if Logo/Stamp model missing.
      }

      final result = await SubmitVerificationService.instance.submit(
        documentImage: widget.documentImage,
        personImage: widget.personImage,
        idType: widget.idType,
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
                ? const Center(child: CircularProgressIndicator())
                : _error != null
                    ? _ErrorView(message: _error!, onRetry: _run)
                    : _ResultView(
                        result: _result!,
                        idType: widget.idType,
                        documentDecision: _documentDecision,
                        documentPercent: _documentPercent,
                      ),
          ),
        ),
      ),
    );
  }
}

class _ErrorView extends StatelessWidget {
  const _ErrorView({required this.message, required this.onRetry});

  final String message;
  final VoidCallback onRetry;

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
    required this.idType,
    required this.documentDecision,
    required this.documentPercent,
  });

  final SubmitVerificationResult result;
  final String idType;
  final String? documentDecision;
  final double? documentPercent;

  @override
  Widget build(BuildContext context) {
    final similarity = result.face.similarityPercent;
    final similarityText = '${similarity.toStringAsFixed(1)}%';
    final isMatch = result.face.match;

    return ListView(
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
                _kv('نوع الهوية', idType),
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
