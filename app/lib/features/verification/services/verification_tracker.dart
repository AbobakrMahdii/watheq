import 'dart:async';

import 'package:flutter/foundation.dart';

import '../../../core/storage/secure_storage_service.dart';
import '../models/verification_models.dart';
import 'notification_service.dart';
import 'verification_orchestrator_service.dart';

/// Global singleton that tracks an active (pending/running) verification.
///
/// • Polls the backend every 2 seconds in a [Timer].
/// • Persists the active ID to secure storage so it survives app restarts.
/// • Shows a local notification when the verification finishes.
/// • Exposes a [ValueNotifier] so the UI can react to changes.
class VerificationTracker extends ChangeNotifier {
  VerificationTracker._();

  static final VerificationTracker instance = VerificationTracker._();

  static const String _storageKey = 'active_verification_id';

  // ── Public observable state ─────────────────────────────────────────
  int? activeId;
  VerificationRecord? lastRecord;
  List<VerificationStep> steps = [];
  VerificationStage? currentStage;

  /// Non-null after verification finishes (success/failed).
  VerificationStatus? finalStatus;

  /// True while polling is active.
  bool get isTracking => _timer != null && _timer!.isActive;

  /// True when there is something to show the user (in-progress or just finished).
  bool get hasActiveVerification => activeId != null;

  // ── Internal ────────────────────────────────────────────────────────
  Timer? _timer;
  bool _polling = false;

  // ── Lifecycle ───────────────────────────────────────────────────────

  /// Call from [main] to resume tracking after an app restart.
  Future<void> resumeIfNeeded() async {
    final stored = await SecureStorageService.instance.read(_storageKey);
    if (stored == null) return;
    final id = int.tryParse(stored);
    if (id == null) return;

    // Check the current status — it may have finished while the app was closed.
    try {
      final record = await VerificationOrchestratorService.instance.getStatus(
        id,
      );
      if (record.status == VerificationStatus.pending ||
          record.status == VerificationStatus.running) {
        activeId = id;
        lastRecord = record;
        currentStage = record.currentStage;
        _startPolling();
        notifyListeners();
      } else {
        // Already finished — show notification, expose result, clear storage.
        activeId = id;
        lastRecord = record;
        finalStatus = record.status;
        steps = await VerificationOrchestratorService.instance.getSteps(id);
        await _clearStorage();
        await NotificationService.instance.showVerificationComplete(
          verificationId: id,
          success: record.status == VerificationStatus.success,
        );
        notifyListeners();
      }
    } catch (_) {
      // API unreachable — clear stale data.
      await _clearStorage();
    }
  }

  /// Start tracking a newly submitted verification.
  Future<void> track(int verificationId) async {
    // Stop any previous tracking.
    _timer?.cancel();
    _polling = false;

    activeId = verificationId;
    finalStatus = null;
    lastRecord = null;
    steps = [];
    currentStage = null;

    await SecureStorageService.instance.write(
      _storageKey,
      verificationId.toString(),
    );
    _startPolling();
    notifyListeners();
  }

  /// Update state from an external source (e.g. the result screen that does its
  /// own polling).  This avoids double-fetching.
  void updateFromExternal(
    VerificationRecord record,
    List<VerificationStep> steps,
  ) {
    lastRecord = record;
    this.steps = steps;
    currentStage = record.currentStage;
    notifyListeners();
  }

  /// Call when the user has acknowledged the result (e.g. opened details).
  Future<void> dismiss() async {
    _timer?.cancel();
    _polling = false;
    activeId = null;
    lastRecord = null;
    steps = [];
    currentStage = null;
    finalStatus = null;
    await _clearStorage();
    notifyListeners();
  }

  // ── Polling ─────────────────────────────────────────────────────────

  void _startPolling() {
    _timer?.cancel();
    _timer = Timer.periodic(const Duration(seconds: 2), (_) => _poll());
  }

  Future<void> _poll() async {
    if (activeId == null || _polling) return;
    _polling = true;

    try {
      final record = await VerificationOrchestratorService.instance.getStatus(
        activeId!,
      );
      final fetchedSteps = await VerificationOrchestratorService.instance
          .getSteps(activeId!);

      lastRecord = record;
      steps = fetchedSteps;
      currentStage = record.currentStage;

      if (record.status == VerificationStatus.success ||
          record.status == VerificationStatus.failed) {
        _timer?.cancel();
        finalStatus = record.status;
        await _clearStorage();

        // Show OS notification.
        await NotificationService.instance.showVerificationComplete(
          verificationId: activeId!,
          success: record.status == VerificationStatus.success,
        );
      }

      notifyListeners();
    } catch (e) {
      debugPrint('VerificationTracker poll error: $e');
    }

    _polling = false;
  }

  Future<void> _clearStorage() async {
    await SecureStorageService.instance.delete(_storageKey);
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }
}
