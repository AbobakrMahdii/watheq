import 'dart:io';

import 'package:flutter/material.dart';

import '../../../core/constants/app_colors.dart';
import '../../../core/constants/app_dimensions.dart';
import '../../../features/verification/models/document_type_model.dart';
import '../../../features/verification/services/document_type_api_service.dart';
import '../../../ui/widgets/app_snackbars.dart';
import '../../camera/document_capture_screen.dart';
import '../../camera/selfie_liveness_screen.dart';
import '../../verification/verification_result_screen.dart';

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
  File? personImage;
  Map<String, dynamic>? _livenessData;

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
      _documentTypes = await DocumentTypeApiService.instance
          .getActiveDocumentTypes();
      if (_documentTypes.isNotEmpty) {
        _selectedDocumentType = _documentTypes.first;
      }
    } catch (e) {
      _documentTypesError = 'فشل تحميل أنواع الوثائق: ${e.toString()}';
      if (mounted) {
        AppSnackbars.error(context, _documentTypesError!);
      }
    } finally {
      if (!mounted) return;
      setState(() {
        _isLoadingDocumentTypes = false;
      });
    }
  }

  // Updated form completion logic
  bool get isFormComplete {
    if (_selectedDocumentType == null ||
        documentImageFront == null ||
        personImage == null) {
      return false;
    }
    return true;
  }

  // Updated openCamera to handle front/person images
  Future<void> openCamera(String imageType) async {
    if (imageType == 'front') {
      final File? result = await Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => const DocumentCaptureScreen(),
          fullscreenDialog: true,
        ),
      );
      if (result != null) {
        setState(() {
          documentImageFront = result;
        });
      }
      return;
    }

    final SelfieCaptureResult? result = await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => const SelfieLivenessScreen(useFrontCamera: true),
        fullscreenDialog: true,
      ),
    );
    if (result != null) {
      setState(() {
        personImage = result.file;
        _livenessData = result.livenessData;
      });
    }
  }

  void _uploadData() {
    _openResults();
  }

  Future<void> _openResults() async {
    if (_isSubmitting) return;
    if (!isFormComplete) return;

    final docFront = documentImageFront;
    final person = personImage;
    final selectedType = _selectedDocumentType;

    if (docFront == null || person == null || selectedType == null) return;

    setState(() => _isSubmitting = true);
    try {
      await Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => VerificationResultScreen(
            documentImageFront: docFront,
            documentImageBack: null,
            personImage: person,
            documentTypeId: selectedType.id,
            documentTypeName: selectedType.name,
            livenessData: _livenessData,
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
      appBar: AppBar(title: const Text('وثيق')),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(AppDimensions.padLg),
          child: Column(
            children: [
              Center(
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 520),
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
                                    });
                                  },
                                ),

                          const SizedBox(height: 16),

                          ElevatedButton(
                            onPressed: () => openCamera('front'),
                            child: const Text('صورة الوثيقة الأمامية'),
                          ),
                          if (documentImageFront != null) ...[
                            const SizedBox(height: 8),
                            Image.file(documentImageFront!, height: 120),
                          ],

                          const SizedBox(height: 16),

                          ElevatedButton(
                            onPressed: () => openCamera('person'),
                            child: const Text('صورة الشخص'),
                          ),
                          if (personImage != null) ...[
                            const SizedBox(height: 8),
                            Image.file(personImage!, height: 120),
                          ],

                          const SizedBox(height: 24),

                          ElevatedButton(
                            onPressed: isFormComplete && !_isSubmitting
                                ? _uploadData
                                : null,
                            style: ElevatedButton.styleFrom(
                              padding: const EdgeInsets.all(14),
                            ),
                            child: Text(
                              _isSubmitting ? 'جاري الرفع...' : 'رفع',
                            ),
                          ),
                        ],
                      ),
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
