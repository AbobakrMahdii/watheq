class IpfsPinResult {
  const IpfsPinResult({
    required this.cid,
    required this.filename,
  });

  final String cid;
  final String filename;

  factory IpfsPinResult.fromJson(Map<String, dynamic> json) {
    return IpfsPinResult(
      cid: (json['cid'] as String?) ?? '',
      filename: (json['filename'] as String?) ?? '',
    );
  }
}

