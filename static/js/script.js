// script.js
// Shared front-end interactions for the Future Interest Analysis dashboard.

document.addEventListener('DOMContentLoaded', function () {
    // Mobile sidebar toggle
    const menuToggle = document.getElementById('menuToggle');
    const sidebar = document.getElementById('sidebar');

    if (menuToggle && sidebar) {
        menuToggle.addEventListener('click', function () {
            sidebar.classList.toggle('open');
        });

        // Close sidebar when a nav link is clicked (mobile UX)
        sidebar.querySelectorAll('.nav-link').forEach(function (link) {
            link.addEventListener('click', function () {
                sidebar.classList.remove('open');
            });
        });
    }

    // Auto-dismiss flash alerts after a few seconds
    document.querySelectorAll('.app-alert').forEach(function (alertEl) {
        setTimeout(function () {
            const bsAlert = bootstrap.Alert.getOrCreateInstance(alertEl);
            if (bsAlert) bsAlert.close();
        }, 6000);
    });
});
