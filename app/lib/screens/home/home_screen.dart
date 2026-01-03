import 'dart:io';

import 'package:flutter/material.dart';

import '../../core/constants/app_colors.dart';
import '../../core/constants/app_dimensions.dart';
import '../../features/auth/services/auth_service.dart';
import '../../ui/widgets/app_snackbars.dart';
import '../camera/camera_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String? selectedIdType;
  File? documentImage;
  File? personImage;

  final List<String> idTypes = ['الهوية الوطنية', 'جواز السفر', 'رخصة القيادة'];

  bool get isFormComplete =>
      selectedIdType != null && documentImage != null && personImage != null;

  Future<void> openCamera(bool isDocument) async {
    final File? result = await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => const CameraScreen(),
        fullscreenDialog: true,
      ),
    );

    if (result != null) {
      setState(() {
        if (isDocument) {
          documentImage = result;
        } else {
          personImage = result;
        }
      });
    }
  }

  Future<void> _logout(BuildContext context) async {
    await AuthService.instance.logout();
    if (!context.mounted) return;
    AppSnackbars.success(context, 'تم تسجيل الخروج');
    Navigator.pushReplacementNamed(context, '/login');
  }

  void _uploadData() {
    // هنا ربط API لاحقًا
    AppSnackbars.success(context, 'تم رفع البيانات بنجاح');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Watheq'),
        actions: [
          IconButton(
            onPressed: () => _logout(context),
            tooltip: 'تسجيل الخروج',
            icon: const Icon(Icons.logout),
          ),
        ],
      ),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 520),
          child: Padding(
            padding: const EdgeInsets.all(AppDimensions.padLg),
            child: Card(
              child: Padding(
                padding: const EdgeInsets.all(AppDimensions.padLg),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    const Icon(
                      Icons.verified_user_outlined,
                      size: 44,
                      color: AppColors.primary,
                    ),
                    const SizedBox(height: 10),
                    const Text(
                      'توثيق الهوية',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.w900,
                        color: AppColors.textPrimary,
                      ),
                    ),
                    const SizedBox(height: 16),

                    // نوع الهوية
                    DropdownButtonFormField<String>(
                      value: selectedIdType,
                      decoration: const InputDecoration(
                        labelText: 'نوع الهوية',
                        border: OutlineInputBorder(),
                      ),
                      items: idTypes
                          .map(
                            (e) => DropdownMenuItem(value: e, child: Text(e)),
                          )
                          .toList(),
                      onChanged: (value) {
                        setState(() {
                          selectedIdType = value;
                        });
                      },
                    ),

                    const SizedBox(height: 16),

                    // تصوير الوثيقة
                    ElevatedButton(
                      onPressed: () => openCamera(true),
                      child: const Text('صورة الوثيقة'),
                    ),
                    if (documentImage != null) ...[
                      const SizedBox(height: 8),
                      Image.file(documentImage!, height: 120),
                    ],

                    const SizedBox(height: 16),

                    // تصوير الشخص
                    ElevatedButton(
                      onPressed: () => openCamera(false),
                      child: const Text('صورة الشخص'),
                    ),
                    if (personImage != null) ...[
                      const SizedBox(height: 8),
                      Image.file(personImage!, height: 120),
                    ],

                    const SizedBox(height: 24),

                    // زر الرفع
                    ElevatedButton(
                      onPressed: isFormComplete ? _uploadData : null,
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.all(14),
                      ),
                      child: const Text('رفع'),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
