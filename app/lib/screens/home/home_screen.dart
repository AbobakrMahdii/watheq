import 'dart:io';

import 'package:flutter/material.dart';

import '../../core/constants/app_colors.dart';
import '../../core/constants/app_dimensions.dart';
import '../../features/auth/services/auth_service.dart';
import '../../features/biometric/services/face_verify_service.dart';
import '../../features/verification/models/document_type_model.dart'; // New Import
import '../../features/verification/services/document_type_api_service.dart'; // New Import
import '../../ui/widgets/app_snackbars.dart';
import '../camera/camera_screen.dart';
import '../verification/verification_result_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  // Document Type related state
  List<DocumentTypeModel> _documentTypes = [];
  DocumentTypeModel? _selectedDocumentType;
  bool _isLoadingDocumentTypes = true;
  String? _documentTypesError;

  // Image files
  File? documentImageFront; // For front image
  File? documentImageBack;  // For back image (optional based on document type)
  File? personImage;

  bool _isVerifying = false;
  bool _isSubmitting = false;

  @override
  void initState() {
    super.initState();
    _loadDocumentTypes();
  }

  Future<void> _loadDocumentTypes() async {
    setState(() {
      _isLoadingDocumentTypes = true;
      _documentTypesError = null;
    });
    try {
      _documentTypes = await DocumentTypeApiService.instance.getActiveDocumentTypes();
      if (_documentTypes.isNotEmpty) {
        _selectedDocumentType = _documentTypes.first;
      }
    } catch (e) {
      _documentTypesError = 'فشل تحميل أنواع الوثائق: ${e.toString()}';
      AppSnackbars.error(context, _documentTypesError!);
    } finally {
      setState(() {
        _isLoadingDocumentTypes = false;
      });
    }
  }

  // Updated form completion logic
  bool get isFormComplete {
    if (_selectedDocumentType == null || documentImageFront == null || personImage == null) {
      return false;
    }
    if (_selectedDocumentType!.requiresBackImage && documentImageBack == null) {
      return false;
    }
    return true;
  }

  // Updated openCamera to handle front/back/person images
  Future<void> openCamera(String imageType) async {
    final File? result = await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => const CameraScreen(),
        fullscreenDialog: true,
      ),
    );

    if (result != null) {
      setState(() {
        if (imageType == 'front') {
          documentImageFront = result;
        } else if (imageType == 'back') {
          documentImageBack = result;
        } else if (imageType == 'person') {
          personImage = result;
        }
      });

      // Trigger face verification only after both document front and person images are captured
      if (documentImageFront != null && personImage != null) {
        await _verifyMatch();
      }
    }
  }

  Future<void> _verifyMatch() async {
    if (_isVerifying) return;
    final doc = documentImageFront; // Use front image for face verification
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
    _openResults();
  }

  Future<void> _openResults() async {
    if (_isSubmitting) return;
    if (!isFormComplete) return;

    final docFront = documentImageFront;
    final docBack = documentImageBack; // Optional back image
    final person = personImage;
    final selectedType = _selectedDocumentType; // Use the selected DocumentTypeModel

    if (docFront == null || person == null || selectedType == null) return;
    // If back image is required but not provided, also return
    if (selectedType.requiresBackImage && docBack == null) return;

    setState(() => _isSubmitting = true);
    try {
      await Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => VerificationResultScreen(
            documentImage: docFront,
            personImage: person,
            // Pass the DocumentTypeModel or its ID
            idType: selectedType.name, // Sending name for now, will adjust VerificationResultScreen later
            documentImageBack: docBack, // Pass back image if exists
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
                    _isLoadingDocumentTypes
                        ? const Center(child: CircularProgressIndicator())
                        : _documentTypesError != null
                            ? Text(
                                _documentTypesError!,
                                style: const TextStyle(color: Colors.red),
                                textAlign: TextAlign.center,
                              )
                            : _documentTypes.isEmpty
                                ? const Text(
                                    'لا توجد أنواع وثائق متاحة. يرجى إضافتها من لوحة التحكم.',
                                    textAlign: TextAlign.center,
                                  )
                                : DropdownButtonFormField<DocumentTypeModel>(
                                    value: _selectedDocumentType,
                                    decoration: const InputDecoration(
                                      labelText: 'نوع الهوية',
                                      border: OutlineInputBorder(),
                                    ),
                                    items: _documentTypes
                                        .map(
                                          (docType) => DropdownMenuItem(
                                            value: docType,
                                            child: Text(docType.name),
                                          ),
                                        )
                                        .toList(),
                                    onChanged: (value) {
                                      setState(() {
                                        _selectedDocumentType = value;
                                        // Clear back image if it's no longer required
                                        if (!(_selectedDocumentType?.requiresBackImage ?? false)) {
                                          documentImageBack = null;
                                        }
                                      });
                                    },
                                  ),

                    const SizedBox(height: 16),

                    // تصوير الوثيقة الأمامية
                    ElevatedButton(
                      onPressed: () => openCamera('front'),
                      child: const Text('صورة الوثيقة الأمامية'),
                    ),
                    if (documentImageFront != null) ...[
                      const SizedBox(height: 8),
                      Image.file(documentImageFront!, height: 120),
                    ],

                    // تصوير الوثيقة الخلفية (شرطي)
                    if (_selectedDocumentType?.requiresBackImage == true) ...[
                      const SizedBox(height: 16),
                      ElevatedButton(
                        onPressed: () => openCamera('back'),
                        child: const Text('صورة الوثيقة الخلفية'),
                      ),
                      if (documentImageBack != null) ...[
                        const SizedBox(height: 8),
                        Image.file(documentImageBack!, height: 120),
                      ],
                    ],

                    const SizedBox(height: 16),

                    // تصوير الشخص
                    ElevatedButton(
                      onPressed: () => openCamera('person'),
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
