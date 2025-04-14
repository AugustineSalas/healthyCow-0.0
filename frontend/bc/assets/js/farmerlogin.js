// Import the functions you need from the SDKs you need
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.4.0/firebase-app.js";
import { getAuth, signInWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/11.4.0/firebase-auth.js";

// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
    apiKey: "AIzaSyAGf3ApSIcJoLUhtQ7EkCFs7iZp2gi9Vxo",
    authDomain: "bovine-care-9c1ee.firebaseapp.com",
    projectId: "bovine-care-9c1ee",
    storageBucket: "bovine-care-9c1ee.firebasestorage.app",
    messagingSenderId: "586715830813",
    appId: "1:586715830813:web:f8a3134311dd8d3f4a04f4"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);


//buttons
const loginButton = document.getElementById("loginButton");

//register button
document.querySelector("form").addEventListener("submit", function (event) {
    event.preventDefault(); // Prevent default form submission

    const email = document.getElementById("email").value;
    const password = document.getElementById("password").value;

    signInWithEmailAndPassword(auth, email, password)
        .then((userCredential) => {
            alert("Login successful!");
            window.location.href = "farmerdashboard.html"; // Redirect to dashboard or home page
        })
        .catch((error) => {
            alert(error.message); // Show login error
        });
});




