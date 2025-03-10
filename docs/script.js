document.addEventListener('DOMContentLoaded', function() {
    const sections = document.querySelectorAll('section');
    const options = {
        threshold: 0.1
    };

    const observer = new IntersectionObserver(function(entries, observer) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, options);

    sections.forEach(section => {
        observer.observe(section);
    });
});

document.querySelectorAll('.accordion-btn, .accordion-item').forEach(item => {
    item.addEventListener('click', function () {
        const panel = this.nextElementSibling;
        const isVisible = panel.style.display === 'block';
        
        // Hide all panels
        document.querySelectorAll('.panel, .panel-item').forEach(p => {
            p.style.display = 'none';
        });
        
        // Toggle the clicked panel
        panel.style.display = isVisible ? 'none' : 'block';
    });
});





