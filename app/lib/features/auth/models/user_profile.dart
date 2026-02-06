class UserProfile {
  const UserProfile({
    required this.id,
    required this.email,
    required this.role,
    this.name,
    this.username,
  });

  final int? id;
  final String? name;
  final String? username;
  final String email;
  final String role;

  factory UserProfile.fromJson(Map<String, dynamic> json) {
    return UserProfile(
      id: (json['id'] as num?)?.toInt(),
      name: json['name'] as String?,
      username: json['username'] as String?,
      email: (json['email'] as String?) ?? '',
      role: (json['role'] as String?) ?? 'user',
    );
  }
}
