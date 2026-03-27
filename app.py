from rt_rename.web import app
from rt_rename.constants import DEFAULT_HOST, DEFAULT_PORT


if __name__ == "__main__":
    app.run(debug=True, host=DEFAULT_HOST, port=DEFAULT_PORT)
