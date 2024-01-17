from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from bs4_search import chain as bs4_search_chain

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


add_routes(app, bs4_search_chain, path="/bs4-search")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
