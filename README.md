# sky-baker

wgpu を使って exr ファイルを KTX2 cubemap に変換します.
すべて Claude 製です.

## ビルド・実行

```sh
cargo run --release -p sky-baker -- \
    --input sky.exr \
    --output rgba16f.ktx2 \
    --face-size 512
```