# Department Management System (DMS)

A PHP 8.3, MySQL, PDO-based Department Management System for XAMPP. The codebase is organized as one continuous application with public pages, authentication, role-based dashboards, student workflows, staff workflows, admin management, assets, SQL, and documentation.

## Stack

- PHP 8.3
- MySQL
- PDO
- HTML5, CSS3, Vanilla JavaScript
- Chart.js
- DomPDF-ready report structure
- Remita-ready payment structure

## Setup on XAMPP

1. Copy this folder to `htdocs/DMS`.
2. Create/import the database:
   - Import `database/schema.sql`
   - Import `database/seed.sql`
3. Check database credentials in `config/connect.php`.
4. Visit `http://localhost/DMS/index.php`.

## Seed Accounts

All seed passwords are `password`.

- Admin: `admin@dms.test`
- Staff: `staff@dms.test`
- Student: `student@dms.test`

## Main Folders

- `config` - database and application bootstrap
- `components` - shared header, footer, sidebar
- `helpers` - escaping, CSRF, flash messages, path helpers
- `middleware` - authentication and role checks
- `services` - reusable application services
- `assets` - CSS and JavaScript
- `admin` - administrator workflows
- `student` - student workflows
- `staff` - staff workflows
- `auth` - login, registration, logout, recovery
- `database` - schema and seed data
- `storage` - uploads, receipts, generated reports
- `docs` - project documentation

## Security Baseline

- Password hashing with `password_hash`
- PDO prepared statements
- CSRF validation for form submissions
- Output escaping helper
- Role-based route guards
- Session timeout
- Upload validation helper
- Audit/activity log tables

## Notes

The Remita and DomPDF integration points are represented structurally and ready for live credentials/packages. Add Remita merchant settings in the `settings` table before enabling live payment verification.
