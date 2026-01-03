class AuthSession {
  const AuthSession({
    required this.accessToken,
    required this.tokenType,
    required this.role,
    required this.email,
  });

  final String accessToken;
  final String tokenType;
  final String role;
  final String email;
}
