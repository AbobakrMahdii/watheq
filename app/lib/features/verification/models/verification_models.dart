import 'package:flutter/material.dart';

/// مراحل التحقق — متطابقة مع VerificationStage في الباك‌إند
enum VerificationStage {
  documentImageQuality,
  documentCropping,
  documentFaceExtraction,
  faceMatching,
  ocr,
  aiVerification,
  dataVerification,
  blockchain,
}

enum VerificationStatus { pending, running, success, failed }

VerificationStage stageFromString(String value) {
  switch (value.toUpperCase()) {
    case 'DOCUMENT_IMAGE_QUALITY':
      return VerificationStage.documentImageQuality;
    case 'DOCUMENT_CROPPING':
      return VerificationStage.documentCropping;
    case 'DOCUMENT_FACE_EXTRACTION':
      return VerificationStage.documentFaceExtraction;
    case 'FACE_MATCHING':
      return VerificationStage.faceMatching;
    case 'OCR':
      return VerificationStage.ocr;
    case 'AI_VERIFICATION':
      return VerificationStage.aiVerification;
    case 'DATA_VERIFICATION':
      return VerificationStage.dataVerification;
    case 'BLOCKCHAIN':
      return VerificationStage.blockchain;
    // Legacy fallbacks
    case 'SELFIE_LIVENESS':
    case 'BIOMETRIC':
      return VerificationStage.faceMatching;
    case 'ML':
      return VerificationStage.aiVerification;
    default:
      return VerificationStage.documentImageQuality;
  }
}

/// تسمية عربية لكل مرحلة تحقق
String stageArabicLabel(VerificationStage stage) {
  switch (stage) {
    case VerificationStage.documentImageQuality:
      return 'جودة الصورة';
    case VerificationStage.documentCropping:
      return 'قص الوثيقة';
    case VerificationStage.documentFaceExtraction:
      return 'استخراج الوجه';
    case VerificationStage.faceMatching:
      return 'مطابقة الوجه';
    case VerificationStage.ocr:
      return 'قراءة النصوص';
    case VerificationStage.aiVerification:
      return 'تحقق الذكاء الاصطناعي';
    case VerificationStage.dataVerification:
      return 'مطابقة البيانات';
    case VerificationStage.blockchain:
      return 'تسجيل البلوكتشين';
  }
}

/// أيقونة لكل مرحلة تحقق
IconData stageIcon(VerificationStage stage) {
  switch (stage) {
    case VerificationStage.documentImageQuality:
      return Icons.image_search;
    case VerificationStage.documentCropping:
      return Icons.crop;
    case VerificationStage.documentFaceExtraction:
      return Icons.face_retouching_natural;
    case VerificationStage.faceMatching:
      return Icons.compare_arrows;
    case VerificationStage.ocr:
      return Icons.text_fields;
    case VerificationStage.aiVerification:
      return Icons.psychology;
    case VerificationStage.dataVerification:
      return Icons.fact_check;
    case VerificationStage.blockchain:
      return Icons.shield;
  }
}

VerificationStatus statusFromString(String value) {
  switch (value.toUpperCase()) {
    case 'RUNNING':
      return VerificationStatus.running;
    case 'SUCCESS':
      return VerificationStatus.success;
    case 'FAILED':
      return VerificationStatus.failed;
    case 'PENDING':
    default:
      return VerificationStatus.pending;
  }
}

class VerificationStep {
  VerificationStep({
    required this.id,
    required this.stage,
    required this.status,
    this.errorMessage,
    this.resultData,
  });

  final int id;
  final VerificationStage stage;
  final VerificationStatus status;
  final String? errorMessage;
  final Map<String, dynamic>? resultData;

  factory VerificationStep.fromJson(Map<String, dynamic> json) {
    return VerificationStep(
      id: (json['id'] as num).toInt(),
      stage: stageFromString((json['stage'] as String?) ?? ''),
      status: statusFromString((json['status'] as String?) ?? ''),
      errorMessage: json['error_message'] as String?,
      resultData: json['result_data'] as Map<String, dynamic>?,
    );
  }
}

class VerificationRecord {
  VerificationRecord({
    required this.id,
    required this.status,
    this.currentStage,
    this.errorMessage,
    this.resultData,
    this.createdAt,
  });

  final int id;
  final VerificationStatus status;
  final VerificationStage? currentStage;
  final String? errorMessage;
  final Map<String, dynamic>? resultData;
  final DateTime? createdAt;

  factory VerificationRecord.fromJson(Map<String, dynamic> json) {
    final currentStageValue = json['current_stage'] as String?;
    return VerificationRecord(
      id: (json['id'] as num).toInt(),
      status: statusFromString((json['status'] as String?) ?? ''),
      currentStage: currentStageValue == null
          ? null
          : stageFromString(currentStageValue),
      errorMessage: json['error_message'] as String?,
      resultData: json['result_data'] as Map<String, dynamic>?,
      createdAt: json['created_at'] != null
          ? DateTime.tryParse(json['created_at'] as String)
          : null,
    );
  }
}

class VerificationSummary {
  const VerificationSummary({
    required this.total,
    required this.success,
    required this.failed,
    required this.running,
    required this.pending,
    required this.items,
  });

  final int total;
  final int success;
  final int failed;
  final int running;
  final int pending;
  final List<VerificationRecord> items;

  factory VerificationSummary.fromJson(Map<String, dynamic> json) {
    final itemsJson = json['items'] as List<dynamic>? ?? [];
    return VerificationSummary(
      total: (json['total'] as num?)?.toInt() ?? 0,
      success: (json['status_counts']?['SUCCESS'] as num?)?.toInt() ?? 0,
      failed: (json['status_counts']?['FAILED'] as num?)?.toInt() ?? 0,
      running: (json['status_counts']?['RUNNING'] as num?)?.toInt() ?? 0,
      pending: (json['status_counts']?['PENDING'] as num?)?.toInt() ?? 0,
      items: itemsJson
          .map((e) => VerificationRecord.fromJson(e as Map<String, dynamic>))
          .toList(),
    );
  }
}
