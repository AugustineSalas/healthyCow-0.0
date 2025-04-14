// Import the functions you need from the SDKs you need
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.4.0/firebase-app.js";
import { getAuth, createUserWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/11.4.0/firebase-auth.js";

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
const registerButton = document.getElementById("registerButton");

//register button
registerButton.addEventListener("click", function (event) {
    event.preventDefault();


    //inputs
    const email = document.getElementById("email").value;
    const password = document.getElementById("password").value;
    createUserWithEmailAndPassword(auth, email, password)
        .then((userCredential) => {
            // Signed up 
            const user = userCredential.user;
            alert("User registered successfully");
            window.location.href = "farmerlogin.html";
            // ...
        })
        .catch((error) => {
            const errorCode = error.code;
            const errorMessage = error.message;
            alert(errorMessage);
            // ..
        });


});
