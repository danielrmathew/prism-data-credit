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
        
        // Toggle the clicked panel
        panel.style.display = panel.style.display === 'block' ? 'none' : 'block';
    });
});
let currentIndex = 0;
const totalItems = document.querySelectorAll('.carousel-item').length;
const carouselItems = document.querySelectorAll('.carousel-item');
const carouselIndicator = document.querySelector('.carousel-indicator');

function moveCarousel(direction) {
    carouselItems[currentIndex].style.display = 'none';
    
    currentIndex = (currentIndex + direction + totalItems) % totalItems;

    carouselItems[currentIndex].style.display = 'block';

    carouselIndicator.textContent = `${currentIndex + 1}/${totalItems}`;
}

carouselItems[currentIndex].style.display = 'block';
carouselIndicator.textContent = `${currentIndex + 1}/${totalItems}`;






