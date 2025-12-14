import streamlit as st
from auth import _read_users, hash_password, verify_password, validate_username, validate_password, _append_user

st.set_page_config(page_title="Multi-Domain Intelligence Dashboard - Login", page_icon="🔐", layout="centered")


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = None



def login_user(username, password):
    users = _read_users()
    if username not in users:
        return False, "Username not found."

    if verify_password(password, users[username]):
        st.session_state.logged_in = True
        st.session_state.username = username
        return True, "Login successful!"
    else:
        return False, "Incorrect password."


def register_user(username, password):
    ok, msg = validate_username(username)
    if not ok:
        return False, msg

    ok, msg = validate_password(password)
    if not ok:
        return False, msg

    users = _read_users()
    if username in users:
        return False, "Username already exists."

    hashed = hash_password(password)
    _append_user(username, hashed)

    return True, "Registration successful! You may now login."


st.title("🔐Multi-Domain Intelligence Dashboard")
st.subheader("Secure Login System")

tab_login, tab_register = st.tabs(["Login", "Register"])



with tab_login:
    st.write("### Login to your account")

    login_username = st.text_input("Username", key="login_username")
    login_pw = st.text_input("Password", type="password", key="login_password")

    if st.button("Login", type="primary"):
        success, message = login_user(login_username.strip(), login_pw.strip())
        if success:
            st.success(message)
            st.rerun()
        else:
            st.error(message)


with tab_register:
    st.write("### Create a new account")

    reg_username = st.text_input("New Username", key="reg_user")
    reg_password = st.text_input("New Password", type="password", key="reg_pw")
    reg_confirm = st.text_input("Confirm Password", type="password", key="reg_pw2")

    if st.button("Register"):
        if reg_password != reg_confirm:
            st.error("Passwords do not match.")
        else:
            success, message = register_user(reg_username.strip(), reg_password.strip())
            if success:
                st.success(message)
            else:
                st.error(message)

# If already logged in, show quick navigation and logout
if st.session_state.logged_in:
    st.success(f"Logged in as **{st.session_state.username}**")
    st.write("Redirecting to dashboard...")
    if st.button("Go to Dashboard"):
        st.query_params["page"] = "dashboard"
        st.rerun()
  
    if st.button("Log out"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()    