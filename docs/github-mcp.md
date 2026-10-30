# GitHub 연결

연구 기록은 GitHub의 가설별 Draft PR에서 관리한다. 계정 연결 성공과 저장소 쓰기 권한을 확인한 뒤 게시한다.

## 공식 로컬 MCP 서버

GitHub 공식 릴리스 `github-mcp-server`의 네이티브 실행 파일을 사용할 수 있다.
공식 배포 파일의 SHA-256을 확인하고 개인 설정에 등록한다. 실제 설치 경로·인증정보는 저장소에 커밋하지 않는다.

```bash
codex mcp add github -- /path/to/github-mcp-server stdio --toolsets repos,issues,pull_requests,actions,users --oauth-scopes repo,workflow
```

이 방식의 OAuth는 GitHub MCP 서버가 처리한다. `codex mcp login`은 HTTP 서버용이므로
이 stdio 설정에는 사용하지 않는다. 서버를 다시 불러오고 첫 GitHub 도구를 사용할 때 브라우저의 GitHub 인증을 완료한다.
기본 OAuth 앱이 포함된 공식 릴리스를 사용하며, 사용자가 토큰을 채팅에 전달할 필요가 없다.
이 로컬 방식은 OAuth 토큰을 서버 프로세스 메모리에만 유지하므로 재시작 후 다시 인증이 필요할 수 있다.

`codex mcp list`에 등록된 것만으로 계정 연결·저장소 접근에 성공했다고 보고하지 않는다.
도구로 현재 사용자, 대상 저장소, 열린 PR을 읽어 연결을 확인하고 실제 Draft PR URL을 확인한다.
PR 작성·파일 변경 기능이 해당 도구에 없으면 인증된 Git/CLI를 보조로 사용한다.
MCP 로그인과 Git/CLI 로그인은 서로 다른 인증 경로이므로 각각 성공 여부를 확인한다.

## 참고

- [GitHub 공식 Codex 설치 안내](https://github.com/github/github-mcp-server/blob/main/docs/installation-guides/install-codex.md)
- [GitHub 로컬 서버 OAuth](https://github.com/github/github-mcp-server/blob/main/docs/oauth-login.md)
- [OpenAI 공식 MCP 설정 안내](https://learn.chatgpt.com/docs/extend/mcp)

이 설정 문서는 연구 계획·결과를 PR에 남기는 규칙을 대신하지 않는다.
