class DocumentTypeModel {
  final int id;
  final String name;
  final bool isActive;
  final bool requiresBackImage;
  final DateTime createdAt;

  DocumentTypeModel({
    required this.id,
    required this.name,
    required this.isActive,
    required this.requiresBackImage,
    required this.createdAt,
  });

  factory DocumentTypeModel.fromJson(Map<String, dynamic> json) {
    return DocumentTypeModel(
      id: json['id'],
      name: json['name'],
      isActive: json['is_active'],
      requiresBackImage: json['requires_back_image'],
      createdAt: DateTime.parse(json['created_at']),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'name': name,
      'is_active': isActive,
      'requires_back_image': requiresBackImage,
      'created_at': createdAt.toIso8601String(),
    };
  }
}
