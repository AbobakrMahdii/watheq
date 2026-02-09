import 'dart:io';

import 'package:flutter/material.dart';

import '../../core/constants/app_dimensions.dart';
import '../../features/auth/services/auth_service.dart';
import '../../ui/widgets/verification_fab.dart';
import '../citizens/citizens_screen.dart';
import 'views/home_screen.dart';
import 'views/profile_screen.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  int _currentIndex = 0;
  final List<Widget> _screens = const [HomeScreen(), ProfileScreen()];

  /// Current user role — loaded once on init.
  String _role = 'user';

  @override
  void initState() {
    super.initState();
    _loadRole();
  }

  Future<void> _loadRole() async {
    final session = await AuthService.instance.getSavedSession();
    if (session != null && mounted) {
      setState(() => _role = session.role);
    }
  }

  bool get _isSuperAdmin => _role == 'super_admin';

  void _showExitConfirmDialog() {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(AppDimensions.radiusMd),
        ),
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.orange.shade50,
                borderRadius: BorderRadius.circular(AppDimensions.radiusSm),
              ),
              child: Icon(
                Icons.exit_to_app_rounded,
                color: Colors.orange.shade700,
                size: 24,
              ),
            ),
            const SizedBox(width: 12),
            const Text(
              'تأكيد الخروج',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
            ),
          ],
        ),
        content: const Padding(
          padding: EdgeInsets.only(top: 8),
          child: Text(
            'هل أنت متأكد أنك تريد الخروج من التطبيق؟',
            style: TextStyle(fontSize: 15, height: 1.5),
          ),
        ),
        actions: [
          OutlinedButton(
            onPressed: () => Navigator.of(ctx).pop(),
            style: OutlinedButton.styleFrom(
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
              side: BorderSide(color: Colors.grey.shade300),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(AppDimensions.radiusSm),
              ),
            ),
            child: const Text(
              'إلغاء',
              style: TextStyle(fontSize: 15, fontWeight: FontWeight.w600),
            ),
          ),
          ElevatedButton.icon(
            onPressed: () => exit(exitCode),
            icon: const Icon(Icons.exit_to_app, size: 18),
            label: const Text(
              'خروج',
              style: TextStyle(fontSize: 15, fontWeight: FontWeight.w600),
            ),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.orange.shade600,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
              elevation: 0,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(AppDimensions.radiusSm),
              ),
            ),
          ),
        ],
        actionsPadding: const EdgeInsets.fromLTRB(24, 12, 24, 20),
        actionsAlignment: MainAxisAlignment.end,
      ),
    );
  }

  // ── Sidebar (super_admin only) ─────────────────────────────────
  Widget _buildDrawer() {
    return Drawer(
      child: SafeArea(
        child: Column(
          children: [
            DrawerHeader(
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.primaryContainer,
              ),
              child: const Row(
                children: [
                  Icon(Icons.admin_panel_settings, size: 36),
                  SizedBox(width: 12),
                  Text(
                    'لوحة المشرف',
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                ],
              ),
            ),
            ListTile(
              leading: const Icon(Icons.people_alt_outlined),
              title: const Text('سجلات المواطنين'),
              subtitle: const Text('عرض وتعديل بيانات المواطنين'),
              onTap: () {
                Navigator.pop(context); // close drawer
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const CitizensScreen()),
                );
              },
            ),
            const Divider(),
            ListTile(
              leading: const Icon(Icons.home_outlined),
              title: const Text('الرئيسية'),
              onTap: () {
                Navigator.pop(context);
                setState(() => _currentIndex = 0);
              },
            ),
            ListTile(
              leading: const Icon(Icons.person_outline),
              title: const Text('الملف الشخصي'),
              onTap: () {
                Navigator.pop(context);
                setState(() => _currentIndex = 1);
              },
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (_, __) async {
        if (_currentIndex != 0) {
          setState(() => _currentIndex = 0);
          return;
        } else {
          _showExitConfirmDialog();
          return;
        }
      },
      child: Scaffold(
        // Drawer only for super_admin
        drawer: _isSuperAdmin ? _buildDrawer() : null,
        appBar: _isSuperAdmin
            ? AppBar(
                title: const Text('وثيق'),
                leading: Builder(
                  builder: (ctx) => IconButton(
                    icon: const Icon(Icons.menu),
                    onPressed: () => Scaffold.of(ctx).openDrawer(),
                  ),
                ),
              )
            : null,
        body: Stack(
          children: [
            IndexedStack(index: _currentIndex, children: _screens),
            const VerificationFab(),
          ],
        ),
        bottomNavigationBar: BottomNavigationBar(
          currentIndex: _currentIndex,
          onTap: (index) => setState(() => _currentIndex = index),
          items: const [
            BottomNavigationBarItem(
              icon: Icon(Icons.home_outlined),
              label: 'الرئيسية',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.person_outline),
              label: 'الملف الشخصي',
            ),
          ],
        ),
      ),
    );
  }
}
