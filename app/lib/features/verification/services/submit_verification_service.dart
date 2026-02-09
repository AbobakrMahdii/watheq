import 'dart:io';

import 'package:crypto/crypto.dart';

import '../../auth/services/auth_service.dart';
import '../../biometric/models/face_verify_result.dart';
import '../../biometric/services/face_verify_service.dart';
import '../models/ipfs_pin_result.dart';
import 'ipfs_service.dart';
import 'ocr_service.dart';

class SubmitVerificationResult {
  const SubmitVerificationResult({
    required this.face,
    required this.ipfs,
    required this.ocr,
    required this.docId,
    required this.sha256,
    this.aiElements = const {},
    this.dataVerificationResult = const {},
  });

  final FaceVerifyResult face;
  final IpfsPinResult ipfs;
  final Map<String, dynamic> ocr;
  final String docId;
  final String sha256;

  /// نتائج تحقق الذكاء الاصطناعي لكل عنصر (logo, stamp, ...)
  final Map<String, dynamic> aiElements;

  /// نتائج مطابقة البيانات مع سجلات المواطنين
  final Map<String, dynamic> dataVerificationResult;
}

class SubmitVerificationService {
  SubmitVerificationService._();

  static final SubmitVerificationService instance =
      SubmitVerificationService._();

  Future<SubmitVerificationResult> submit({
    required File documentImageFront,
    File? documentImageBack,
    required File personImage,
    required int documentTypeId,
  }) async {
    final session = await AuthService.instance.getSavedSession();
    if (session == null) {
      throw const SubmitVerificationException('غير مسجل الدخول');
    }

    final FaceVerifyResult face;
    try {
      face = await FaceVerifyService.instance.verify(
        documentPhoto: documentImageFront,
        personPhoto: personImage,
      );
    } catch (e) {
      throw SubmitVerificationException(
        'فشل مطابقة الوجه: ${e.toString()}',
        cause: e,
      );
    }
    if (!face.match) {
      throw SubmitVerificationException(
        'فشل التطابق (التشابه ${face.similarityPercent.toStringAsFixed(1)}%)',
      );
    }

    final String sha;
    try {
      final bytes = await documentImageFront.readAsBytes();
      sha = sha256.convert(bytes).toString();
    } catch (e) {
      throw SubmitVerificationException(
        'فشل حساب بصمة الملف (SHA256)',
        cause: e,
      );
    }

    final IpfsPinResult ipfs;
    try {
      ipfs = await IpfsService.instance.pinFile(file: documentImageFront);
    } catch (e) {
      throw SubmitVerificationException(
        'فشل رفع الوثيقة إلى IPFS: ${e.toString()}',
        cause: e,
      );
    }

    final Map<String, dynamic> ocr;
    try {
      ocr = await OcrService.instance.ocrDocument(file: documentImageFront);
    } catch (e) {
      throw SubmitVerificationException(
        'فشل قراءة OCR: ${e.toString()}',
        cause: e,
      );
    }

    final docId =
        'DOC-${DateTime.now().millisecondsSinceEpoch}-${documentTypeId}';
    // Blockchain recording is now handled server-side in the verification pipeline

    return SubmitVerificationResult(
      face: face,
      ipfs: ipfs,
      ocr: ocr,
      docId: docId,
      sha256: sha,
    );
  }
}

class SubmitVerificationException implements Exception {
  const SubmitVerificationException(this.message, {this.cause});

  final String message;
  final Object? cause;

  @override
  String toString() => message;
}
