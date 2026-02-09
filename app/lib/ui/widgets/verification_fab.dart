import 'package:flutter/material.dart';

import '../../core/constants/app_colors.dart';
import '../../features/verification/models/verification_models.dart';
import '../../features/verification/services/verification_tracker.dart';
import '../../screens/verification/verification_details_screen.dart';

/// A draggable floating action button that appears when a verification is
/// actively being processed (or just finished).
///
/// Place this inside a [Stack] that sits above the page content (e.g. via
/// [MaterialApp.builder]).
class VerificationFab extends StatefulWidget {
  const VerificationFab({super.key});

  @override
  State<VerificationFab> createState() => _VerificationFabState();
}

class _VerificationFabState extends State<VerificationFab>
    with SingleTickerProviderStateMixin {
  late final AnimationController _pulse;
  Offset _offset = const Offset(16, 120);

  @override
  void initState() {
    super.initState();
    _pulse = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _pulse.dispose();
    super.dispose();
  }

  void _navigateToDetails(BuildContext context) {
    final tracker = VerificationTracker.instance;
    final id = tracker.activeId;
    if (id == null) return;

    // If still running, push the progress screen (result screen is gone).
    // If finished, go to details.
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => VerificationDetailsScreen(verificationId: id),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: VerificationTracker.instance,
      builder: (context, _) {
        final tracker = VerificationTracker.instance;
        if (!tracker.hasActiveVerification) {
          return const SizedBox.shrink();
        }

        final isDone = tracker.finalStatus != null;
        final isSuccess = tracker.finalStatus == VerificationStatus.success;
        final isFailed = tracker.finalStatus == VerificationStatus.failed;

        Color bgColor;
        IconData icon;
        String tooltip;

        if (isDone && isSuccess) {
          bgColor = AppColors.success;
          icon = Icons.check_circle;
          tooltip = 'التحقق اكتمل بنجاح — اضغط للتفاصيل';
        } else if (isDone && isFailed) {
          bgColor = AppColors.danger;
          icon = Icons.error;
          tooltip = 'فشل التحقق — اضغط للتفاصيل';
        } else {
          bgColor = AppColors.primary;
          icon = Icons.hourglass_top;
          tooltip = 'جاري التحقق... اضغط للعودة';
        }

        return Positioned(
          left: _offset.dx,
          top: _offset.dy,
          child: GestureDetector(
            onPanUpdate: (details) {
              setState(() {
                _offset += details.delta;
              });
            },
            child: AnimatedBuilder(
              animation: _pulse,
              builder: (context, child) {
                final scale = isDone ? 1.0 : 1.0 + _pulse.value * 0.08;
                return Transform.scale(scale: scale, child: child);
              },
              child: Material(
                elevation: 6,
                shape: const CircleBorder(),
                color: bgColor,
                child: InkWell(
                  customBorder: const CircleBorder(),
                  onTap: () => _navigateToDetails(context),
                  child: Container(
                    width: 56,
                    height: 56,
                    alignment: Alignment.center,
                    child: isDone
                        ? Icon(icon, color: Colors.white, size: 28)
                        : Stack(
                            alignment: Alignment.center,
                            children: [
                              const SizedBox(
                                width: 34,
                                height: 34,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2.5,
                                  color: Colors.white,
                                ),
                              ),
                              Text(
                                '${tracker.steps.where((s) => s.status == VerificationStatus.success).length}/${tracker.steps.length}',
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 10,
                                  fontWeight: FontWeight.w800,
                                ),
                              ),
                            ],
                          ),
                  ),
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}
