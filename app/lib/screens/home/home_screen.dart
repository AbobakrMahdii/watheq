import 'dart:io';

import 'package:flutter/material.dart';

import '../../core/constants/app_colors.dart';
import '../../core/constants/app_dimensions.dart';
import '../../features/auth/services/auth_service.dart';
import '../../features/biometric/services/face_verify_service.dart';
import '../../ui/widgets/app_snackbars.dart';
import '../camera/camera_screen.dart';
import '../verification/verification_result_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String? selectedIdType;
  File? documentImage;
  File? personImage;
  bool _isVerifying = false;
  bool _isSubmitting = false;

  final List<String> idTypes = ['الهوية الوطنية', 'جواز السفر', 'رخصة القيادة'];

  @override
  void initState() {
    super.initState();
    selectedIdType ??= idTypes.first;
  }

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

      if (documentImage != null && personImage != null) {
        await _verifyMatch();
      }
    }
  }

  Future<void> _verifyMatch() async {
    if (_isVerifying) return;
    final doc = documentImage;
    final person = personImage;
    if (doc == null || person == null) return;

    setState(() => _isVerifying = true);

    try {
      var dialogShown = false;
      showDialog<void>(
        context: context,
        barrierDismissible: false,
        builder: (_) => const Center(child: CircularProgressIndicator()),
      ).then((_) => dialogShown = false);
      dialogShown = true;

      final result = await FaceVerifyService.instance.verify(
        documentPhoto: doc,
        personPhoto: person,
      );

      if (!mounted) return;
      if (dialogShown && Navigator.of(context).canPop()) {
        Navigator.of(context).pop();
      }

      final similarity = result.similarityPercent.toStringAsFixed(1);
      if (result.match) {
        AppSnackbars.success(context, 'تم التطابق بنسبة $similarity%');
      } else {
        AppSnackbars.error(context, 'لا يوجد تطابق (التشابه $similarity%)');
      }
    } catch (e) {
      if (!mounted) return;
      if (Navigator.of(context).canPop()) {
        Navigator.of(context).pop();
      }
      final message = e is FaceVerifyException ? e.message : e.toString();
      AppSnackbars.error(context, message);
    } finally {
      if (mounted) setState(() => _isVerifying = false);
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
    _openResults();
  }

  Future<void> _openResults() async {
    if (_isSubmitting) return;
    if (!isFormComplete) return;

    final doc = documentImage;
    final person = personImage;
    final idType = selectedIdType;
    if (doc == null || person == null || idType == null) return;

    setState(() => _isSubmitting = true);
    try {
      await Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => VerificationResultScreen(
            documentImage: doc,
            personImage: person,
            idType: idType,
          ),
          fullscreenDialog: true,
        ),
      );
    } finally {
      if (mounted) setState(() => _isSubmitting = false);
    }
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
                      initialValue: selectedIdType,
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
                      onPressed:
                          isFormComplete && !_isSubmitting ? _uploadData : null,
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.all(14),
                      ),
                      child: Text(_isSubmitting ? 'جاري الرفع...' : 'رفع'),
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
