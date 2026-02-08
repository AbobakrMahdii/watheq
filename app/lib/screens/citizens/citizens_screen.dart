import 'package:flutter/material.dart';

import '../../../features/citizens/services/citizen_service.dart';

/// Full-screen page that lists all citizen records (super_admin only).
/// Tapping a row opens an edit dialog.
class CitizensScreen extends StatefulWidget {
  const CitizensScreen({super.key});

  @override
  State<CitizensScreen> createState() => _CitizensScreenState();
}

class _CitizensScreenState extends State<CitizensScreen> {
  final _service = CitizenService.instance;
  List<Map<String, dynamic>> _citizens = [];
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final list = await _service.listCitizens(limit: 200);
      if (mounted) setState(() => _citizens = list);
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  // ── Labels (Arabic) ──────────────────────────────────────────────
  static const _fieldLabels = <String, String>{
    'national_id': 'رقم الهوية',
    'full_name_ar': 'الاسم بالعربي',
    'full_name_en': 'الاسم بالإنجليزي',
    'date_of_birth': 'تاريخ الميلاد',
    'address': 'العنوان',
    'issue_date': 'تاريخ الإصدار',
    'expiry_date': 'تاريخ الانتهاء',
    'gender': 'الجنس',
    'nationality': 'الجنسية',
    'document_type': 'نوع الوثيقة',
  };

  static const _editableFields = [
    'full_name_ar',
    'full_name_en',
    'date_of_birth',
    'address',
    'issue_date',
    'expiry_date',
    'gender',
    'nationality',
    'document_type',
  ];

  // ── Edit dialog ──────────────────────────────────────────────────
  Future<void> _showEditDialog(Map<String, dynamic> citizen) async {
    final controllers = <String, TextEditingController>{};
    for (final key in _editableFields) {
      controllers[key] = TextEditingController(
        text: citizen[key]?.toString() ?? '',
      );
    }

    final saved = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(
          'تعديل: ${citizen['national_id'] ?? ''}',
          style: const TextStyle(fontSize: 16),
        ),
        content: SizedBox(
          width: double.maxFinite,
          child: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: _editableFields.map((key) {
                return Padding(
                  padding: const EdgeInsets.symmetric(vertical: 6),
                  child: TextField(
                    controller: controllers[key],
                    decoration: InputDecoration(
                      labelText: _fieldLabels[key] ?? key,
                      border: const OutlineInputBorder(),
                      isDense: true,
                    ),
                  ),
                );
              }).toList(),
            ),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('إلغاء'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('حفظ'),
          ),
        ],
      ),
    );

    if (saved != true) return;

    final updates = <String, String?>{};
    for (final key in _editableFields) {
      final newVal = controllers[key]!.text.trim();
      final oldVal = (citizen[key]?.toString() ?? '').trim();
      if (newVal != oldVal) updates[key] = newVal;
    }
    for (final c in controllers.values) {
      c.dispose();
    }

    if (updates.isEmpty) return;

    try {
      await _service.updateCitizen(citizen['national_id'], updates);
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(const SnackBar(content: Text('تم التحديث بنجاح')));
      }
      await _load();
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('فشل التحديث: $e')));
      }
    }
  }

  // ── Delete confirm ───────────────────────────────────────────────
  Future<void> _confirmDelete(Map<String, dynamic> citizen) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('تأكيد الحذف'),
        content: Text(
          'هل أنت متأكد من حذف سجل ${citizen['full_name_ar'] ?? citizen['national_id']}؟',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('إلغاء'),
          ),
          FilledButton(
            style: FilledButton.styleFrom(backgroundColor: Colors.red),
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('حذف'),
          ),
        ],
      ),
    );
    if (confirmed != true) return;

    try {
      await _service.deleteCitizen(citizen['national_id']);
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(const SnackBar(content: Text('تم الحذف')));
      }
      await _load();
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('فشل الحذف: $e')));
      }
    }
  }

  // ── Build ────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('سجلات المواطنين')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
          ? Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(_error!, textAlign: TextAlign.center),
                  const SizedBox(height: 12),
                  ElevatedButton(
                    onPressed: _load,
                    child: const Text('إعادة المحاولة'),
                  ),
                ],
              ),
            )
          : _citizens.isEmpty
          ? const Center(child: Text('لا توجد سجلات بعد'))
          : RefreshIndicator(
              onRefresh: _load,
              child: ListView.separated(
                padding: const EdgeInsets.all(12),
                itemCount: _citizens.length,
                separatorBuilder: (_, __) => const Divider(height: 1),
                itemBuilder: (_, i) {
                  final c = _citizens[i];
                  return _CitizenTile(
                    citizen: c,
                    labels: _fieldLabels,
                    onEdit: () => _showEditDialog(c),
                    onDelete: () => _confirmDelete(c),
                  );
                },
              ),
            ),
    );
  }
}

// ── Citizen list tile ──────────────────────────────────────────────
class _CitizenTile extends StatelessWidget {
  const _CitizenTile({
    required this.citizen,
    required this.labels,
    required this.onEdit,
    required this.onDelete,
  });

  final Map<String, dynamic> citizen;
  final Map<String, String> labels;
  final VoidCallback onEdit;
  final VoidCallback onDelete;

  @override
  Widget build(BuildContext context) {
    final nid = citizen['national_id'] ?? '-';
    final name = citizen['full_name_ar'] ?? citizen['full_name_en'] ?? '-';
    final dob = citizen['date_of_birth'] ?? '';
    final nationality = citizen['nationality'] ?? '';

    return Card(
      margin: const EdgeInsets.symmetric(vertical: 4),
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: onEdit,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
          child: Row(
            children: [
              CircleAvatar(
                backgroundColor: Theme.of(context).colorScheme.primaryContainer,
                child: const Icon(Icons.person),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      name,
                      style: const TextStyle(
                        fontWeight: FontWeight.w600,
                        fontSize: 15,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      'الهوية: $nid',
                      style: TextStyle(
                        color: Colors.grey.shade600,
                        fontSize: 13,
                      ),
                    ),
                    if (dob.isNotEmpty || nationality.isNotEmpty)
                      Text(
                        [
                          if (dob.isNotEmpty) dob,
                          if (nationality.isNotEmpty) nationality,
                        ].join(' • '),
                        style: TextStyle(
                          color: Colors.grey.shade500,
                          fontSize: 12,
                        ),
                      ),
                  ],
                ),
              ),
              IconButton(
                icon: const Icon(Icons.edit_outlined, size: 20),
                onPressed: onEdit,
                tooltip: 'تعديل',
              ),
              IconButton(
                icon: Icon(
                  Icons.delete_outline,
                  size: 20,
                  color: Colors.red.shade400,
                ),
                onPressed: onDelete,
                tooltip: 'حذف',
              ),
            ],
          ),
        ),
      ),
    );
  }
}
