import 'package:flutter/material.dart';

import '../../../core/constants/app_colors.dart';
import '../../../core/constants/app_dimensions.dart';
import '../../../features/verification/models/verification_models.dart';
import '../../../features/verification/services/verification_orchestrator_service.dart';
import '../../../ui/widgets/app_snackbars.dart';
import '../../verification/verification_details_screen.dart';

class VerificationHistoryScreen extends StatefulWidget {
  const VerificationHistoryScreen({super.key});

  @override
  State<VerificationHistoryScreen> createState() =>
      _VerificationHistoryScreenState();
}

class _VerificationHistoryScreenState extends State<VerificationHistoryScreen> {
  bool _loading = true;
  VerificationSummary? _summary;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() => _loading = true);
    try {
      final summary = await VerificationOrchestratorService.instance.listMy(
        pageSize: 50,
      );
      if (!mounted) return;
      setState(() => _summary = summary);
    } catch (e) {
      if (mounted) AppSnackbars.error(context, e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('سجل عمليات التحقق'),
        actions: [
          IconButton(
            onPressed: _load,
            icon: const Icon(Icons.refresh),
            tooltip: 'تحديث',
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _summary == null
          ? const Center(child: Text('لا يوجد بيانات متاحة.'))
          : Padding(
              padding: const EdgeInsets.all(AppDimensions.padLg),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Wrap(
                    spacing: 10,
                    runSpacing: 10,
                    children: [
                      _StatChip(label: 'الإجمالي', value: _summary!.total),
                      _StatChip(
                        label: 'ناجحة',
                        value: _summary!.success,
                        color: Colors.green,
                      ),
                      _StatChip(
                        label: 'قيد التنفيذ',
                        value: _summary!.running + _summary!.pending,
                        color: Colors.orange,
                      ),
                      _StatChip(
                        label: 'فاشلة',
                        value: _summary!.failed,
                        color: Colors.redAccent,
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  Expanded(
                    child: _summary!.items.isEmpty
                        ? const Center(child: Text('لا يوجد سجل تحقق بعد.'))
                        : ListView.separated(
                            itemCount: _summary!.items.length,
                            separatorBuilder: (_, __) => const Divider(),
                            itemBuilder: (_, index) {
                              final v = _summary!.items[index];
                              final statusColor =
                                  v.status == VerificationStatus.success
                                  ? Colors.green
                                  : (v.status == VerificationStatus.failed
                                        ? Colors.redAccent
                                        : Colors.orange);
                              return ListTile(
                                leading: Icon(
                                  Icons.verified,
                                  color: statusColor,
                                ),
                                title: Text('التحقق #${v.id}'),
                                subtitle: Text(
                                  'الحالة: ${v.status.name.toUpperCase()}'
                                  '${v.createdAt != null ? " • ${v.createdAt}" : ""}',
                                ),
                                trailing: const Icon(
                                  Icons.arrow_forward_ios,
                                  size: 14,
                                ),
                                onTap: () {
                                  Navigator.push(
                                    context,
                                    MaterialPageRoute(
                                      builder: (_) => VerificationDetailsScreen(
                                        verificationId: v.id,
                                      ),
                                    ),
                                  );
                                },
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

class _StatChip extends StatelessWidget {
  const _StatChip({required this.label, required this.value, this.color});

  final String label;
  final int value;
  final Color? color;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: (color ?? Colors.blueGrey.shade50).withOpacity(0.25),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color ?? Colors.blueGrey.shade200),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: TextStyle(fontSize: 12, color: color ?? Colors.blueGrey),
          ),
          const SizedBox(height: 2),
          Text(
            value.toString(),
            style: TextStyle(
              fontWeight: FontWeight.w800,
              fontSize: 16,
              color: color ?? Colors.black87,
            ),
          ),
        ],
      ),
    );
  }
}
