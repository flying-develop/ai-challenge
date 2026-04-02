"""
Роутер вызовов инструментов.
Парсит [TOOL_CALL] блоки из ответа LLM и выполняет соответствующие методы.
"""
from __future__ import annotations

import re

from llm_agent.file_tools import FileSystemToolkit


class ToolExecutor:
    """
    Парсит вызов инструмента из ответа LLM и выполняет его.
    LLM возвращает вызов в формате [TOOL_CALL]...[/TOOL_CALL].
    """

    # ─── Парсинг ──────────────────────────────────────────────────────────────

    def parse_tool_call(self, llm_response: str) -> dict | None:
        """
        Парсит блок:
            [TOOL_CALL]
            name: search_in_files
            query: Editor
            directory: src/
            [/TOOL_CALL]

        Поддерживает многострочные значения (например, content для write_file).
        Возвращает: {name: str, param1: str, ...} или None если блока нет.
        """
        match = re.search(r"\[TOOL_CALL\]\s*(.*?)\[/TOOL_CALL\]", llm_response, re.DOTALL)
        if not match:
            return None
        return self._parse_kv_block(match.group(1))

    def parse_finish(self, llm_response: str) -> dict | None:
        """
        Парсит блок:
            [FINISH]
            summary: что было сделано
            files_read: файл1, файл2
            files_modified:
            [/FINISH]

        Возвращает: {summary: str, files_read: str, files_modified: str} или None.
        """
        match = re.search(r"\[FINISH\]\s*(.*?)\[/FINISH\]", llm_response, re.DOTALL)
        if not match:
            return None
        return self._parse_kv_block(match.group(1))

    # ─── Выполнение ───────────────────────────────────────────────────────────

    def execute(self, tool_call: dict, toolkit: FileSystemToolkit) -> str:
        """
        Роутер: по tool_call['name'] вызвать нужный метод toolkit.
        При ошибке возвращает описание ошибки (не бросает исключение).
        """
        name = tool_call.get("name", "")
        params = {k: v for k, v in tool_call.items() if k != "name"}
        try:
            result = self._dispatch(name, params, toolkit)
            return self.format_result(name, result)
        except Exception as exc:
            return self.format_result(name, {"error": str(exc)})

    def _dispatch(self, name: str, params: dict, toolkit: FileSystemToolkit):
        if name == "list_files":
            return toolkit.list_files(
                pattern=params.get("pattern", "*"),
                base_dir=params.get("base_dir", "."),
            )

        if name == "read_file":
            return toolkit.read_file(
                path=params["path"],
                start_line=int(params.get("start_line") or 1),
                end_line=int(params["end_line"]) if params.get("end_line") else None,
            )

        if name == "search_in_files":
            return toolkit.search_in_files(
                query=params["query"],
                directory=params.get("directory", "."),
                file_pattern=params.get("file_pattern", "*"),
                case_sensitive=str(params.get("case_sensitive", "false")).lower() == "true",
            )

        if name == "write_file":
            return toolkit.write_file(
                path=params["path"],
                content=params.get("content", ""),
                create_dirs=str(params.get("create_dirs", "true")).lower() != "false",
            )

        if name == "apply_diff":
            return toolkit.apply_diff(
                path=params["path"],
                old_text=params.get("old_text", ""),
                new_text=params.get("new_text", ""),
            )

        if name == "show_diff":
            return toolkit.show_diff(
                path=params["path"],
                old_text=params.get("old_text", ""),
                new_text=params.get("new_text", ""),
            )

        if name == "finish":
            return params  # возвращаем как есть, agent_loop его перехватит

        raise ValueError(f"Неизвестный инструмент: {name!r}")

    # ─── Форматирование ───────────────────────────────────────────────────────

    def format_result(self, tool_name: str, result) -> str:
        """
        Форматировать результат инструмента для LLM.

        Формат:
            [TOOL_RESULT: {tool_name}]
            {result}
            [/TOOL_RESULT]
        """
        body = self._render(tool_name, result)
        return f"[TOOL_RESULT: {tool_name}]\n{body}\n[/TOOL_RESULT]"

    def _render(self, tool_name: str, result) -> str:
        if isinstance(result, str):
            return result

        if isinstance(result, dict):
            if "error" in result:
                return f"Ошибка: {result['error']}"
            return self._render_dict(tool_name, result)

        if isinstance(result, list):
            return self._render_list(tool_name, result)

        return str(result)

    def _render_dict(self, tool_name: str, result: dict) -> str:
        if tool_name == "read_file":
            lines = []
            if result.get("warning"):
                lines.append(f"⚠️  {result['warning']}")
            lines.append(
                f"Файл: {result['path']} "
                f"(строки {result['shown_lines']} из {result['total_lines']})"
            )
            lines.append("─" * 40)
            lines.append(result.get("content", ""))
            return "\n".join(lines)

        if tool_name == "write_file":
            parts = [f"✅ Записан: {result['path']}"]
            parts.append(f"   Байт записано: {result['bytes_written']}")
            if result.get("backup_created"):
                parts.append(f"   Резервная копия: {result['path']}.bak")
            return "\n".join(parts)

        if tool_name == "apply_diff":
            parts = [f"✅ Изменён: {result['path']}"]
            parts.append(f"   Замен сделано: {result['replacements_made']}")
            if result.get("backup_created"):
                parts.append(f"   Резервная копия: {result['path']}.bak")
            if result.get("warning"):
                parts.append(f"   ⚠️  {result['warning']}")
            return "\n".join(parts)

        # Generic key: value
        return "\n".join(f"{k}: {v}" for k, v in result.items() if v)

    def _render_list(self, tool_name: str, items: list) -> str:
        if not items:
            return "Результатов не найдено."

        if not isinstance(items[0], dict):
            return "\n".join(str(i) for i in items)

        first = items[0]

        # search_in_files results
        if "file_path" in first:
            lines = [f"Найдено {len(items)} вхождений:\n"]
            for item in items:
                lines.append(f"{item['file_path']}:{item['line_number']}")
                if item.get("context_before"):
                    lines.append(f"  до:    {item['context_before']}")
                lines.append(f"  >>> {item['line_content']}")
                if item.get("context_after"):
                    lines.append(f"  после: {item['context_after']}")
                lines.append("")
            return "\n".join(lines)

        # list_files results
        if "path" in first and "size_bytes" in first:
            lines = [f"Найдено {len(items)} файлов:"]
            for item in items:
                size_kb = item["size_bytes"] / 1024
                lines.append(f"  {item['path']}  ({size_kb:.1f} KB{item['extension']})")
            return "\n".join(lines)

        # Fallback
        return "\n".join(str(i) for i in items)

    # ─── Вспомогательные ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_kv_block(block: str) -> dict:
        """Парсит блок key: value (поддерживает многострочные значения).

        Автоматически снимает обрамляющие кавычки с однострочных значений,
        чтобы обрабатывать паттерны вида pattern: "*.tsx" → *.tsx.
        """
        result: dict[str, str] = {}
        current_key: str | None = None
        current_lines: list[str] = []

        for line in block.split("\n"):
            kv = re.match(r"^(\w+):\s*(.*)", line)
            if kv:
                if current_key is not None:
                    result[current_key] = ToolExecutor._finalize_value(current_lines)
                current_key = kv.group(1)
                current_lines = [kv.group(2)]
            elif current_key is not None:
                current_lines.append(line)

        if current_key is not None:
            result[current_key] = ToolExecutor._finalize_value(current_lines)

        return result

    @staticmethod
    def _finalize_value(lines: list[str]) -> str:
        """Собрать значение из строк, для однострочных — снять кавычки."""
        raw = "\n".join(lines).strip()
        # Снимаем обрамляющие кавычки только для однострочных значений
        if "\n" not in raw and len(raw) >= 2:
            if (raw[0] == '"' and raw[-1] == '"') or (raw[0] == "'" and raw[-1] == "'"):
                return raw[1:-1]
        return raw
