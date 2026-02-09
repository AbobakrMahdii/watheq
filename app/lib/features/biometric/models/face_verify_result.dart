class FaceVerifyResult {
  const FaceVerifyResult({
    required this.match,
    required this.similarityPercent,
    required this.acceptThresholdPercent,
  });

  /// Whether the face was accepted (similarity >= threshold)
  final bool match;
  final double similarityPercent;
  final double acceptThresholdPercent;

  factory FaceVerifyResult.fromJson(Map<String, dynamic> json) {
    return FaceVerifyResult(
      match: (json['accepted'] as bool?) ?? false,
      similarityPercent: (json['similarity_percent'] as num?)?.toDouble() ?? 0,
      acceptThresholdPercent:
          (json['accept_threshold_percent'] as num?)?.toDouble() ?? 80,
    );
  }
}
