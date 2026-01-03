class UserModel {
  final int id;
  final String name;
  final String username;
  final String email;

  UserModel({
    required this.id,
    required this.name,
    required this.username,
    required this.email,
  });

  factory UserModel.fromJson(Map<String, dynamic> json) {
    return UserModel(
      id: json['id'] as int,
      name: json['name'] as String,
      username: json['username'] as String,
      email: json['email'] as String,
    );
  }

  Map<String, dynamic> toJson() {
    return {'id': id, 'name': name, 'username': username, 'email': email};
  }

  @override
  String toString() {
    return 'UserModel{id: $id, name: $name, username: $username, email: $email}';
  }
}
