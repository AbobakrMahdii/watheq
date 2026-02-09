import 'package:flutter/material.dart';

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
      builder: (context) => AlertDialog(
        title: const Text('تأكيد الخروج'),
        content: const Text('هل أنت متأكد أنك تريد الخروج من التطبيق؟'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('إلغاء'),
          ),
          TextButton(
            onPressed: () {
              Navigator.of(context).pop();
              Navigator.of(context).maybePop();
            },
            child: const Text('خروج'),
          ),
        ],
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
                title: const Text('وثّق'),
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
