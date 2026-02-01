class DocumentVerifyResult {
  const DocumentVerifyResult({
    required this.finalDecision,
    required this.authenticityPercent,
    required this.logoDecision,
    required this.stampDecision,
  });

  final String finalDecision;
  final double? authenticityPercent;
  final String? logoDecision;
  final String? stampDecision;

  factory DocumentVerifyResult.fromJson(Map<String, dynamic> json) {
    final logo = (json['logo'] as Map?)?.cast<String, dynamic>();
    final stamp = (json['stamp'] as Map?)?.cast<String, dynamic>();
    return DocumentVerifyResult(
      finalDecision: (json['final_decision'] as String?) ?? '',
      authenticityPercent:
          (json['authenticity_percent'] as num?)?.toDouble(),
      logoDecision: (logo?['decision'] as String?),
      stampDecision: (stamp?['decision'] as String?),
    );
  }
}

