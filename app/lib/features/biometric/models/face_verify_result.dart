class FaceVerifyResult {
  const FaceVerifyResult({
    required this.match,
    required this.similarityPercent,
    required this.distance,
    required this.threshold,
    required this.model,
    required this.distanceMetric,
  });

  final bool match;
  final double similarityPercent;
  final double distance;
  final double threshold;
  final String model;
  final String distanceMetric;

  factory FaceVerifyResult.fromJson(Map<String, dynamic> json) {
    return FaceVerifyResult(
      match: (json['match'] as bool?) ?? false,
      similarityPercent: (json['similarity_percent'] as num?)?.toDouble() ?? 0,
      distance: (json['distance'] as num?)?.toDouble() ?? 0,
      threshold: (json['threshold'] as num?)?.toDouble() ?? 0,
      model: (json['model'] as String?) ?? '',
      distanceMetric: (json['distance_metric'] as String?) ?? '',
    );
  }
}

