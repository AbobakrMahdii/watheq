import 'package:flutter/material.dart';

import '../../core/constants/app_colors.dart';
import '../../core/constants/app_dimensions.dart';

class AppSnackbars {
  const AppSnackbars._();

  static void error(BuildContext context, String message) {
    _show(context, message, background: AppColors.danger);
  }

  static void success(BuildContext context, String message) {
    _show(context, message, background: AppColors.success);
  }

  static void info(BuildContext context, String message) {
    _show(context, message, background: AppColors.warning);
  }

  static void _show(
    BuildContext context,
    String message, {
    required Color background,
  }) {
    final snackBar = SnackBar(
      behavior: SnackBarBehavior.floating,
      margin: const EdgeInsets.all(AppDimensions.padMd),
      backgroundColor: background,
      content: Text(
        message,
        style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600),
      ),
    );

    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(snackBar);
  }
}