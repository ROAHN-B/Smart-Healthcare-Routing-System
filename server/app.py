import uvicorn

from backend.api.main import app 

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)
if __name__ == "__main__":
    main()