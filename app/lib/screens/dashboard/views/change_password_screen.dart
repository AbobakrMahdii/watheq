import 'package:flutter/material.dart';

import '../../../core/constants/app_colors.dart';
import '../../../core/constants/app_dimensions.dart';
import '../../../features/auth/services/auth_service.dart';
import '../../../ui/widgets/app_snackbars.dart';

class ChangePasswordScreen extends StatefulWidget {
  const ChangePasswordScreen({super.key});

  @override
  State<ChangePasswordScreen> createState() => _ChangePasswordScreenState();
}

class _ChangePasswordScreenState extends State<ChangePasswordScreen> {
  final _formKey = GlobalKey<FormState>();
  final _currentPasswordCtrl = TextEditingController();
  final _newPasswordCtrl = TextEditingController();
  final _confirmPasswordCtrl = TextEditingController();
  bool _loading = false;
  bool _obscureCurrent = true;
  bool _obscureNew = true;
  bool _obscureConfirm = true;

  @override
  void dispose() {
    _currentPasswordCtrl.dispose();
    _newPasswordCtrl.dispose();
    _confirmPasswordCtrl.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() => _loading = true);
    try {
      await AuthService.instance.changePassword(
        currentPassword: _currentPasswordCtrl.text,
        newPassword: _newPasswordCtrl.text,
      );
      if (!mounted) return;
      AppSnackbars.success(context, 'تم تغيير كلمة المرور بنجاح');
      Navigator.pop(context);
    } on AuthException catch (e) {
      if (!mounted) return;
      AppSnackbars.error(context, e.message);
    } catch (e) {
      if (!mounted) return;
      AppSnackbars.error(context, 'حدث خطأ غير متوقع');
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('تغيير كلمة المرور')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(AppDimensions.padLg),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(AppDimensions.padLg),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'أدخل كلمة المرور الحالية وكلمة المرور الجديدة',
                        style: TextStyle(fontSize: 14, color: Colors.grey),
                      ),
                      const SizedBox(height: AppDimensions.padLg),

                      // Current password
                      TextFormField(
                        controller: _currentPasswordCtrl,
                        obscureText: _obscureCurrent,
                        decoration: InputDecoration(
                          labelText: 'كلمة المرور الحالية',
                          prefixIcon: const Icon(Icons.lock_outline),
                          suffixIcon: IconButton(
                            icon: Icon(
                              _obscureCurrent
                                  ? Icons.visibility_off
                                  : Icons.visibility,
                            ),
                            onPressed: () {
                              setState(
                                () => _obscureCurrent = !_obscureCurrent,
                              );
                            },
                          ),
                          border: const OutlineInputBorder(),
                        ),
                        validator: (v) {
                          if (v == null || v.isEmpty) {
                            return 'كلمة المرور الحالية مطلوبة';
                          }
                          return null;
                        },
                      ),
                      const SizedBox(height: AppDimensions.padMd),

                      // New password
                      TextFormField(
                        controller: _newPasswordCtrl,
                        obscureText: _obscureNew,
                        decoration: InputDecoration(
                          labelText: 'كلمة المرور الجديدة',
                          prefixIcon: const Icon(Icons.lock),
                          suffixIcon: IconButton(
                            icon: Icon(
                              _obscureNew
                                  ? Icons.visibility_off
                                  : Icons.visibility,
                            ),
                            onPressed: () {
                              setState(() => _obscureNew = !_obscureNew);
                            },
                          ),
                          border: const OutlineInputBorder(),
                        ),
                        validator: (v) {
                          if (v == null || v.isEmpty) {
                            return 'كلمة المرور الجديدة مطلوبة';
                          }
                          if (v.length < 6) {
                            return 'كلمة المرور يجب أن تكون 6 أحرف على الأقل';
                          }
                          return null;
                        },
                      ),
                      const SizedBox(height: AppDimensions.padMd),

                      // Confirm new password
                      TextFormField(
                        controller: _confirmPasswordCtrl,
                        obscureText: _obscureConfirm,
                        decoration: InputDecoration(
                          labelText: 'تأكيد كلمة المرور الجديدة',
                          prefixIcon: const Icon(Icons.lock),
                          suffixIcon: IconButton(
                            icon: Icon(
                              _obscureConfirm
                                  ? Icons.visibility_off
                                  : Icons.visibility,
                            ),
                            onPressed: () {
                              setState(
                                () => _obscureConfirm = !_obscureConfirm,
                              );
                            },
                          ),
                          border: const OutlineInputBorder(),
                        ),
                        validator: (v) {
                          if (v == null || v.isEmpty) {
                            return 'تأكيد كلمة المرور مطلوب';
                          }
                          if (v != _newPasswordCtrl.text) {
                            return 'كلمات المرور الجديدة غير متطابقة';
                          }
                          return null;
                        },
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: AppDimensions.padLg),

              SizedBox(
                height: 48,
                child: ElevatedButton(
                  onPressed: _loading ? null : _submit,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.primary,
                    foregroundColor: AppColors.onPrimary,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(
                        AppDimensions.radiusSm,
                      ),
                    ),
                  ),
                  child: _loading
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Colors.white,
                          ),
                        )
                      : const Text(
                          'تغيير كلمة المرور',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
