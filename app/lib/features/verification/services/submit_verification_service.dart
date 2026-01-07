import 'dart:io';

import 'package:crypto/crypto.dart';

import '../../auth/services/auth_service.dart';
import '../../biometric/models/face_verify_result.dart';
import '../../biometric/services/face_verify_service.dart';
import '../models/ipfs_pin_result.dart';
import 'ipfs_service.dart';
import 'ledger_service.dart';
import 'ocr_service.dart';

class SubmitVerificationResult {
  const SubmitVerificationResult({
    required this.face,
    required this.ipfs,
    required this.ocr,
    required this.docId,
    required this.sha256,
    required this.ledgerRecorded,
  });

  final FaceVerifyResult face;
  final IpfsPinResult ipfs;
  final Map<String, dynamic> ocr;
  final String docId;
  final String sha256;
  final bool ledgerRecorded;
}

class SubmitVerificationService {
  SubmitVerificationService._();

  static final SubmitVerificationService instance = SubmitVerificationService._();

  Future<SubmitVerificationResult> submit({
    required File documentImage,
    required File personImage,
    required String idType,
  }) async {
    final session = await AuthService.instance.getSavedSession();
    if (session == null) {
      throw const SubmitVerificationException('غير مسجل الدخول');
    }

    final FaceVerifyResult face;
    try {
      face = await FaceVerifyService.instance.verify(
        documentPhoto: documentImage,
        personPhoto: personImage,
      );
    } catch (e) {
      throw SubmitVerificationException('فشل مطابقة الوجه: ${e.toString()}',
          cause: e);
    }
    if (!face.match) {
      throw SubmitVerificationException(
        'فشل التطابق (التشابه ${face.similarityPercent.toStringAsFixed(1)}%)',
      );
    }

    final String sha;
    try {
      final bytes = await documentImage.readAsBytes();
      sha = sha256.convert(bytes).toString();
    } catch (e) {
      throw SubmitVerificationException('فشل حساب بصمة الملف (SHA256)', cause: e);
    }

    final IpfsPinResult ipfs;
    try {
      ipfs = await IpfsService.instance.pinFile(file: documentImage);
    } catch (e) {
      throw SubmitVerificationException('فشل رفع الوثيقة إلى IPFS: ${e.toString()}',
          cause: e);
    }

    final Map<String, dynamic> ocr;
    try {
      ocr = await OcrService.instance.ocrDocument(file: documentImage);
    } catch (e) {
      throw SubmitVerificationException('فشل قراءة OCR: ${e.toString()}', cause: e);
    }

    final docId =
        'DOC-${DateTime.now().millisecondsSinceEpoch}-${idType.hashCode.abs()}';
    try {
      await LedgerService.instance.createDoc(
        docId: docId,
        cid: ipfs.cid,
        filename: ipfs.filename.isEmpty ? 'document.jpg' : ipfs.filename,
        owner: session.email,
        sha256: sha,
      );
    } catch (e) {
      throw SubmitVerificationException('فشل تسجيل الوثيقة في السجل: ${e.toString()}',
          cause: e);
    }

    return SubmitVerificationResult(
      face: face,
      ipfs: ipfs,
      ocr: ocr,
      docId: docId,
      sha256: sha,
      ledgerRecorded: true,
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
