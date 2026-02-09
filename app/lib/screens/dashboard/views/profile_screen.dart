import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import '../../../core/constants/app_colors.dart';
import '../../../core/services/connection_config_service.dart';
import '../../../core/constants/app_dimensions.dart';
import '../../../features/auth/models/user_profile.dart';
import '../../../features/auth/services/auth_service.dart';
import '../../../ui/widgets/app_snackbars.dart';
import 'change_password_screen.dart';
import 'verification_history_screen.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  bool _loading = true;
  UserProfile? _profile;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() => _loading = true);
    try {
      final profile = await AuthService.instance.fetchProfile();
      if (!mounted) return;
      setState(() {
        _profile = profile;
      });
    } catch (e) {
      if (mounted) AppSnackbars.error(context, e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _logout() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('تسجيل الخروج'),
        content: const Text('هل أنت متأكد من أنك تريد تسجيل الخروج؟'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('إلغاء'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: FilledButton.styleFrom(backgroundColor: Colors.red),
            child: const Text('تسجيل الخروج'),
          ),
        ],
      ),
    );
    if (confirmed != true || !mounted) return;
    await AuthService.instance.logout();
    if (!mounted) return;
    Navigator.pushNamedAndRemoveUntil(context, '/login', (_) => false);
  }

  bool _isAdmin() {
    final role = _profile?.role ?? '';
    return role == 'super_admin' || role == 'admin';
  }

  Future<void> _openAdminDashboard() async {
    final baseUrl = ConnectionConfigService.instance.baseUrl;
    final dashboardUrl = '$baseUrl/dashboard';
    final uri = Uri.parse(dashboardUrl);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    } else {
      if (mounted) {
        AppSnackbars.error(context, 'تعذر فتح لوحة التحكم');
      }
    }
  }

  String _avatarLetters() {
    final text = _profile?.name?.trim().isNotEmpty == true
        ? _profile!.name!
        : (_profile?.email ?? '');
    if (text.isEmpty) return '?';
    final parts = text.split(' ');
    if (parts.length >= 2) {
      return '${parts[0][0]}${parts[1][0]}'.toUpperCase();
    }
    return text[0].toUpperCase();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('الملف الشخصي'),
        actions: [
          IconButton(
            onPressed: _load,
            icon: const Icon(Icons.refresh),
            tooltip: 'تحديث',
          ),
          IconButton(
            onPressed: _logout,
            icon: const Icon(Icons.logout),
            tooltip: 'تسجيل الخروج',
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              padding: const EdgeInsets.all(AppDimensions.padLg),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(AppDimensions.padLg),
                      child: Row(
                        children: [
                          CircleAvatar(
                            radius: 30,
                            backgroundColor: AppColors.primary.withOpacity(
                              0.12,
                            ),
                            child: Text(
                              _avatarLetters(),
                              style: const TextStyle(
                                fontWeight: FontWeight.bold,
                                color: AppColors.primary,
                              ),
                            ),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  _profile?.name ?? _profile?.email ?? 'مستخدم',
                                  style: const TextStyle(
                                    fontSize: 18,
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                                Text(
                                  _profile?.email ?? '—',
                                  style: const TextStyle(
                                    color: Colors.grey,
                                    fontSize: 12,
                                  ),
                                ),
                                Text('الدور: ${_profile?.role ?? '—'}'),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: AppDimensions.padLg),
                  if (_isAdmin())
                    Card(
                      child: ListTile(
                        leading: const Icon(
                          Icons.admin_panel_settings,
                          color: AppColors.primary,
                        ),
                        title: const Text(
                          'لوحة تحكم المشرف',
                          style: TextStyle(
                            fontWeight: FontWeight.w700,
                            fontSize: 15,
                          ),
                        ),
                        subtitle: const Text(
                          'افتح لوحة التحكم الإدارية في المتصفح.',
                          style: TextStyle(fontSize: 12),
                        ),
                        trailing: const Icon(Icons.open_in_browser, size: 16),
                        onTap: _openAdminDashboard,
                      ),
                    ),
                  if (_isAdmin()) const SizedBox(height: AppDimensions.padSm),
                  Card(
                    child: ListTile(
                      leading: const Icon(
                        Icons.verified_user_outlined,
                        color: AppColors.primary,
                      ),
                      title: const Text(
                        'سجل عمليات التحقق',
                        style: TextStyle(
                          fontWeight: FontWeight.w700,
                          fontSize: 15,
                        ),
                      ),
                      subtitle: const Text(
                        'اعرض كل عمليات التحقق الخاصة بك مع إحصاءات تفصيلية.',
                        style: TextStyle(fontSize: 12),
                      ),
                      trailing: const Icon(Icons.arrow_forward_ios, size: 16),
                      onTap: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (_) => const VerificationHistoryScreen(),
                          ),
                        );
                      },
                    ),
                  ),
                  const SizedBox(height: AppDimensions.padSm),
                  Card(
                    child: ListTile(
                      leading: const Icon(
                        Icons.lock_outline,
                        color: AppColors.primary,
                      ),
                      title: const Text(
                        'تغيير كلمة المرور',
                        style: TextStyle(
                          fontWeight: FontWeight.w700,
                          fontSize: 15,
                        ),
                      ),
                      subtitle: const Text(
                        'قم بتحديث كلمة المرور الخاصة بك لحماية حسابك.',
                        style: TextStyle(fontSize: 12),
                      ),
                      trailing: const Icon(Icons.arrow_forward_ios, size: 16),
                      onTap: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (_) => const ChangePasswordScreen(),
                          ),
                        );
                      },
                    ),
                  ),
                ],
              ),
            ),
    );
  }
}
