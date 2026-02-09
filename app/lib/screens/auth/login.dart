import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../core/constants/app_colors.dart';
import '../../core/constants/app_dimensions.dart';
import '../../core/services/connection_config_service.dart';
import '../../features/auth/services/auth_service.dart';
import '../../ui/widgets/app_snackbars.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen>
    with SingleTickerProviderStateMixin {
  final _formKey = GlobalKey<FormState>();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  final _ipController = TextEditingController();
  final _portController = TextEditingController();
  final _dashboardPortController = TextEditingController();

  bool _isLoading = false;
  bool _obscurePassword = true;
  bool _showConnectionSettings = false;

  late final AnimationController _expandController;
  late final Animation<double> _expandAnimation;

  @override
  void initState() {
    super.initState();
    final config = ConnectionConfigService.instance;
    _ipController.text = config.ip;
    _portController.text = config.port;
    _dashboardPortController.text = config.dashboardPort;

    _expandController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 300),
    );
    _expandAnimation = CurvedAnimation(
      parent: _expandController,
      curve: Curves.easeInOut,
    );
  }

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    _ipController.dispose();
    _portController.dispose();
    _dashboardPortController.dispose();
    _expandController.dispose();
    super.dispose();
  }

  void _toggleConnectionSettings() {
    setState(() {
      _showConnectionSettings = !_showConnectionSettings;
      if (_showConnectionSettings) {
        _expandController.forward();
      } else {
        _expandController.reverse();
      }
    });
  }

  Future<void> _submit() async {
    if (_isLoading) return;
    if (!(_formKey.currentState?.validate() ?? false)) return;

    // Apply connection settings before login attempt
    ConnectionConfigService.instance.setConnection(
      ip: _ipController.text.trim(),
      port: _portController.text.trim(),
      dashboardPort: _dashboardPortController.text.trim(),
    );

    setState(() => _isLoading = true);
    try {
      await AuthService.instance.login(
        email: _emailController.text,
        password: _passwordController.text,
      );
      if (!mounted) return;

      // Cache connection settings only on successful login
      await ConnectionConfigService.instance.persist();
      if (!mounted) return;

      Navigator.pushReplacementNamed(context, '/');
    } on AuthException catch (e) {
      if (!mounted) return;
      AppSnackbars.error(context, e.message);
    } catch (_) {
      if (!mounted) return;
      AppSnackbars.error(context, 'تعذر تسجيل الدخول. حاول مرة أخرى.');
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 440),
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(
                horizontal: AppDimensions.padLg,
                vertical: AppDimensions.padMd,
              ),
              child: Form(
                key: _formKey,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    const SizedBox(height: 16),

                    // ── Logo ──
                    Center(
                      child: Container(
                        width: 88,
                        height: 88,
                        decoration: BoxDecoration(
                          color: AppColors.surface,
                          borderRadius: BorderRadius.circular(
                            AppDimensions.radiusLg,
                          ),
                          border: Border.all(color: const Color(0xFFE2E8F0)),
                          boxShadow: [
                            BoxShadow(
                              color: AppColors.primary.withValues(alpha: 0.08),
                              blurRadius: 24,
                              offset: const Offset(0, 8),
                            ),
                          ],
                        ),
                        padding: const EdgeInsets.all(8),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(
                            AppDimensions.radiusMd,
                          ),
                          child: Image.asset(
                            'assets/logo.jpeg',
                            fit: BoxFit.cover,
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(height: 20),

                    // ── Title ──
                    const Text(
                      'مرحباً بك',
                      style: TextStyle(
                        fontSize: 26,
                        fontWeight: FontWeight.w900,
                        color: AppColors.textPrimary,
                        letterSpacing: -0.5,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 6),
                    const Text(
                      'سجّل دخولك للوصول إلى لوحة التحكم',
                      style: TextStyle(
                        color: AppColors.textSecondary,
                        fontSize: 14,
                        height: 1.4,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 24),

                    // ── Connection Settings (collapsible) ──
                    _buildConnectionSettingsCard(),
                    const SizedBox(height: 12),

                    // ── Login Card ──
                    _buildLoginCard(),
                    const SizedBox(height: 16),

                    // ── Footer ──
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          Icons.info_outline_rounded,
                          size: 14,
                          color: AppColors.textSecondary.withValues(alpha: 0.6),
                        ),
                        const SizedBox(width: 6),
                        Text(
                          'إنشاء الحسابات يتم بواسطة مسؤول النظام فقط',
                          style: TextStyle(
                            color: AppColors.textSecondary.withValues(
                              alpha: 0.7,
                            ),
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  // ─────────────────────────────────────────────────────────────────
  //  Connection Settings Card
  // ─────────────────────────────────────────────────────────────────
  Widget _buildConnectionSettingsCard() {
    return Card(
      child: Column(
        children: [
          // Header (always visible)
          InkWell(
            borderRadius: BorderRadius.circular(AppDimensions.radiusLg),
            onTap: _toggleConnectionSettings,
            child: Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: AppDimensions.padMd,
                vertical: AppDimensions.padSm + 2,
              ),
              child: Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: AppColors.primary.withValues(alpha: 0.08),
                      borderRadius: BorderRadius.circular(
                        AppDimensions.radiusSm,
                      ),
                    ),
                    child: Icon(
                      Icons.dns_rounded,
                      size: 18,
                      color: AppColors.primary,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'إعدادات الاتصال',
                          style: TextStyle(
                            fontWeight: FontWeight.w600,
                            fontSize: 14,
                            color: AppColors.textPrimary,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          '${_ipController.text}:${_portController.text}',
                          style: TextStyle(
                            fontSize: 12,
                            color: AppColors.textSecondary.withValues(
                              alpha: 0.8,
                            ),
                            fontFamily: 'monospace',
                          ),
                        ),
                      ],
                    ),
                  ),
                  AnimatedRotation(
                    turns: _showConnectionSettings ? 0.5 : 0,
                    duration: const Duration(milliseconds: 300),
                    child: Icon(
                      Icons.expand_more_rounded,
                      color: AppColors.textSecondary,
                    ),
                  ),
                ],
              ),
            ),
          ),
          // Expandable body
          SizeTransition(
            sizeFactor: _expandAnimation,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(
                AppDimensions.padMd,
                0,
                AppDimensions.padMd,
                AppDimensions.padMd,
              ),
              child: Column(
                children: [
                  const Divider(height: 1),
                  const SizedBox(height: AppDimensions.padMd),
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // IP Address
                      Expanded(
                        flex: 3,
                        child: TextFormField(
                          controller: _ipController,
                          keyboardType: const TextInputType.numberWithOptions(
                            decimal: true,
                          ),
                          inputFormatters: [
                            FilteringTextInputFormatter.allow(RegExp(r'[\d.]')),
                          ],
                          style: const TextStyle(
                            fontFamily: 'monospace',
                            fontSize: 14,
                            letterSpacing: 0.5,
                          ),
                          decoration: InputDecoration(
                            labelText: 'عنوان IP',
                            hintText: '192.168.1.1',
                            hintStyle: TextStyle(
                              color: AppColors.textSecondary.withValues(
                                alpha: 0.4,
                              ),
                              fontFamily: 'monospace',
                            ),
                            prefixIcon: const Icon(
                              Icons.language_rounded,
                              size: 20,
                            ),
                            isDense: true,
                          ),
                          onChanged: (_) => setState(() {}),
                          validator: (value) {
                            final v = (value ?? '').trim();
                            if (v.isEmpty) return 'مطلوب';
                            final parts = v.split('.');
                            if (parts.length != 4) return 'IP غير صالح';
                            for (final p in parts) {
                              final n = int.tryParse(p);
                              if (n == null || n < 0 || n > 255) {
                                return 'IP غير صالح';
                              }
                            }
                            return null;
                          },
                        ),
                      ),
                      const SizedBox(width: 10),
                      // Port
                      Expanded(
                        flex: 2,
                        child: TextFormField(
                          controller: _portController,
                          keyboardType: TextInputType.number,
                          inputFormatters: [
                            FilteringTextInputFormatter.digitsOnly,
                            LengthLimitingTextInputFormatter(5),
                          ],
                          style: const TextStyle(
                            fontFamily: 'monospace',
                            fontSize: 14,
                            letterSpacing: 0.5,
                          ),
                          decoration: InputDecoration(
                            labelText: 'المنفذ',
                            hintText: '8012',
                            hintStyle: TextStyle(
                              color: AppColors.textSecondary.withValues(
                                alpha: 0.4,
                              ),
                              fontFamily: 'monospace',
                            ),
                            prefixIcon: const Icon(Icons.tag, size: 20),
                            isDense: true,
                          ),
                          onChanged: (_) => setState(() {}),
                          validator: (value) {
                            final v = (value ?? '').trim();
                            if (v.isEmpty) return 'مطلوب';
                            final n = int.tryParse(v);
                            if (n == null || n < 1 || n > 65535) {
                              return 'منفذ غير صالح';
                            }
                            return null;
                          },
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  // Dashboard Port
                  TextFormField(
                    controller: _dashboardPortController,
                    keyboardType: TextInputType.number,
                    inputFormatters: [
                      FilteringTextInputFormatter.digitsOnly,
                      LengthLimitingTextInputFormatter(5),
                    ],
                    style: const TextStyle(
                      fontFamily: 'monospace',
                      fontSize: 14,
                      letterSpacing: 0.5,
                    ),
                    decoration: InputDecoration(
                      labelText: 'منفذ لوحة التحكم',
                      hintText: '3200',
                      hintStyle: TextStyle(
                        color: AppColors.textSecondary.withValues(alpha: 0.4),
                        fontFamily: 'monospace',
                      ),
                      prefixIcon: const Icon(Icons.dashboard, size: 20),
                      isDense: true,
                    ),
                    onChanged: (_) => setState(() {}),
                    validator: (value) {
                      final v = (value ?? '').trim();
                      if (v.isEmpty) return 'مطلوب';
                      final n = int.tryParse(v);
                      if (n == null || n < 1 || n > 65535) {
                        return 'منفذ غير صالح';
                      }
                      return null;
                    },
                  ),
                  const SizedBox(height: 10),
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 8,
                    ),
                    decoration: BoxDecoration(
                      color: AppColors.primary.withValues(alpha: 0.04),
                      borderRadius: BorderRadius.circular(
                        AppDimensions.radiusSm,
                      ),
                      border: Border.all(
                        color: AppColors.primary.withValues(alpha: 0.12),
                      ),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Icon(
                              Icons.link_rounded,
                              size: 14,
                              color: AppColors.primary.withValues(alpha: 0.7),
                            ),
                            const SizedBox(width: 8),
                            Text(
                              'API:',
                              style: TextStyle(
                                fontSize: 11,
                                color: AppColors.textSecondary,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                            const SizedBox(width: 4),
                            Expanded(
                              child: Text(
                                'http://${_ipController.text}:${_portController.text}',
                                style: TextStyle(
                                  fontFamily: 'monospace',
                                  fontSize: 11,
                                  color: AppColors.primary.withValues(
                                    alpha: 0.8,
                                  ),
                                  fontWeight: FontWeight.w500,
                                ),
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 4),
                        Row(
                          children: [
                            Icon(
                              Icons.dashboard,
                              size: 14,
                              color: AppColors.primary.withValues(alpha: 0.7),
                            ),
                            const SizedBox(width: 8),
                            Text(
                              'Dashboard:',
                              style: TextStyle(
                                fontSize: 11,
                                color: AppColors.textSecondary,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                            const SizedBox(width: 4),
                            Expanded(
                              child: Text(
                                'http://${_ipController.text}:${_dashboardPortController.text}',
                                style: TextStyle(
                                  fontFamily: 'monospace',
                                  fontSize: 11,
                                  color: AppColors.primary.withValues(
                                    alpha: 0.8,
                                  ),
                                  fontWeight: FontWeight.w500,
                                ),
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ─────────────────────────────────────────────────────────────────
  //  Login Card
  // ─────────────────────────────────────────────────────────────────
  Widget _buildLoginCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(AppDimensions.padLg),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Section label
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: AppColors.primary.withValues(alpha: 0.08),
                    borderRadius: BorderRadius.circular(AppDimensions.radiusSm),
                  ),
                  child: Icon(
                    Icons.lock_outline_rounded,
                    size: 18,
                    color: AppColors.primary,
                  ),
                ),
                const SizedBox(width: 12),
                const Text(
                  'بيانات الدخول',
                  style: TextStyle(
                    fontWeight: FontWeight.w600,
                    fontSize: 14,
                    color: AppColors.textPrimary,
                  ),
                ),
              ],
            ),
            const SizedBox(height: AppDimensions.padMd),

            // Email field
            TextFormField(
              controller: _emailController,
              keyboardType: TextInputType.emailAddress,
              textInputAction: TextInputAction.next,
              autofillHints: const [
                AutofillHints.username,
                AutofillHints.email,
              ],
              decoration: const InputDecoration(
                labelText: 'البريد الإلكتروني',
                prefixIcon: Icon(Icons.alternate_email_rounded),
              ),
              validator: (value) {
                final v = (value ?? '').trim();
                if (v.isEmpty) return 'أدخل البريد الإلكتروني';
                if (!v.contains('@')) return 'بريد إلكتروني غير صالح';
                return null;
              },
            ),
            const SizedBox(height: AppDimensions.padMd),

            // Password field
            TextFormField(
              controller: _passwordController,
              obscureText: _obscurePassword,
              textInputAction: TextInputAction.done,
              autofillHints: const [AutofillHints.password],
              decoration: InputDecoration(
                labelText: 'كلمة المرور',
                prefixIcon: const Icon(Icons.password_rounded),
                suffixIcon: IconButton(
                  onPressed: () =>
                      setState(() => _obscurePassword = !_obscurePassword),
                  icon: Icon(
                    _obscurePassword
                        ? Icons.visibility_rounded
                        : Icons.visibility_off_rounded,
                  ),
                ),
              ),
              onFieldSubmitted: (_) => _submit(),
              validator: (value) {
                if ((value ?? '').isEmpty) return 'أدخل كلمة المرور';
                return null;
              },
            ),
            const SizedBox(height: AppDimensions.padLg),

            // Submit button
            FilledButton(
              onPressed: _isLoading ? null : _submit,
              child: AnimatedSwitcher(
                duration: const Duration(milliseconds: 200),
                child: _isLoading
                    ? const SizedBox(
                        key: ValueKey('loading'),
                        width: 22,
                        height: 22,
                        child: CircularProgressIndicator(
                          strokeWidth: 2.6,
                          color: Colors.white,
                        ),
                      )
                    : const Row(
                        key: ValueKey('text'),
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.login_rounded, size: 20),
                          SizedBox(width: 8),
                          Text('تسجيل الدخول'),
                        ],
                      ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
