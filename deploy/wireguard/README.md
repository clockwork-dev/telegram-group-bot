# WireGuard Client Setup & Operations (`192.168.1.164`)

This directory contains the WireGuard client template and optional `nftables` kill-switch for `192.168.1.164` (Debian 13 Trixie unprivileged Proxmox LXC).

> **Important:** `wg-quick@wg0` is intentionally **disabled** and `/etc/wireguard/wg0.conf` does **not** exist until real VPN keys and endpoint details are populated.

---

## 1. Server Network Baseline (`192.168.1.164`)

- **Interface**: `eth0` (`192.168.1.164/24`, DHCP via `/etc/network/interfaces`)
- **Default Gateway & LAN DNS**: `192.168.1.1` (`/etc/resolv.conf`)
- **LAN Link Route**: `192.168.1.0/24 dev eth0 proto kernel scope link src 192.168.1.164`
- **IPv6 Sysctls**: `/etc/sysctl.d/99-disable-ipv6.conf` sets `net.ipv6.conf.{all,default,lo}.disable_ipv6 = 1`
- **Firewall**: `nftables v1.1.3` (`/etc/nftables.conf` defines `table inet filter` with `accept` policies)
- **DNS Resolver Helpers**: `openresolv` / `resolvconf` / `systemd-resolved` are **not** installed by default

---

## 2. Key Design Decisions in `wg0.conf.example`

### 2.1 `AllowedIPs = 0.0.0.0/0` (IPv4-Only Default) vs `0.0.0.0/0, ::/0`
- Because `net.ipv6.conf.*.disable_ipv6 = 1` is active on `192.168.1.164`, adding `::/0` to `AllowedIPs` or an IPv6 `Address` causes `wg-quick up wg0` to abort when running `ip -6` (`RTNETLINK answers: Permission denied`).
- With `disable_ipv6 = 1`, IPv6 is already disabled kernel-wide (no IPv6 leak).
- **To enable dual-stack IPv4 + IPv6 later**:
  1. Re-enable IPv6 in `/etc/sysctl.d/99-disable-ipv6.conf` and apply:
     ```bash
     sysctl -w net.ipv6.conf.all.disable_ipv6=0 \
               net.ipv6.conf.default.disable_ipv6=0 \
               net.ipv6.conf.eth0.disable_ipv6=0
     ```
  2. Set `Address = <CLIENT_VPN_IP/32>, <CLIENT_VPN_IPV6/128>` in `[Interface]`.
  3. Set `AllowedIPs = 0.0.0.0/0, ::/0` in `[Peer]`.

### 2.2 Mandatory LAN & SSH Preservation (`192.168.1.0/24`)
- `wg-quick` (`Table = auto`) places `default dev wg0` in table `51820` and adds `table main suppress_prefixlength 0`.
- Naive `ip route add 192.168.1.0/24 ...` in `PreUp` fails with `RTNETLINK answers: File exists`, and `ip route del 192.168.1.0/24` in `PreDown` destroys the kernel link route and cuts off SSH.
- Instead, `wg0.conf.example` uses:
  - `PreUp`: Verifies `192.168.1.0/24 dev eth0` exists (`ip -4 route show ... | grep -q ...`).
  - `PostUp`: Idempotently adds `ip -4 rule add to 192.168.1.0/24 lookup main priority 100` (above `wg-quick` rules) so LAN and SSH return traffic always stay on `eth0`.
  - `PreDown`: Removes **only** the `priority 100` policy rule (`|| true`), leaving `eth0`'s link route intact.

### 2.3 DNS Handling (`192.168.1.1` vs VPN DNS)
- `# DNS = <VPN_DNS_IP>` is commented out by default. If you uncomment `DNS =` without installing `openresolv`, `wg-quick up wg0` fails with `resolvconf: command not found`.
- To route DNS through the tunnel instead of `192.168.1.1`:
  ```bash
  apt-get update && apt-get install -y --no-install-recommends openresolv
  ```
  Then uncomment `DNS = <VPN_DNS_IP>` in `/etc/wireguard/wg0.conf` (and optionally uncomment the `192.168.1.1` port-53 `drop` rules in the kill-switch).

### 2.4 Optional `nftables` Kill-Switch (`table inet wg_killswitch`)
- Both inline `PostUp`/`PostDown` rules (commented in `wg0.conf.example`) and the standalone `killswitch.nft.example` file use an isolated `table inet wg_killswitch`.
- It **never** runs `flush ruleset` and **never** modifies `/etc/nftables.conf` (`table inet filter`) or `wg-quick`'s `table ip wg-quick-wg0`.
- Traffic permitted when enabled:
  - Loopback (`lo`) and `established,related` connections
  - Tunnel interface (`wg0`)
  - LAN & SSH (`192.168.1.0/24` on `eth0`)
  - DHCP renewals (`udp sport 68 udp dport 67` / `67 -> 68`)
  - Optional LAN DNS (`192.168.1.1:53`, or blocked via commented `drop` rules when VPN DNS is enabled)
  - Encrypted WireGuard outer UDP packets via `meta mark $(wg show %i fwmark)` (or explicit `ip daddr <VPN_ENDPOINT_IP> udp dport <VPN_ENDPOINT_PORT>`)

---

## 3. Safe Activation, Testing & Rollback (Run as `root` on `192.168.1.164`)

Perform these steps **only after** you have real WireGuard keys and endpoint details:

1. **Create `/etc/wireguard/wg0.conf` with `0600` permissions**:
   ```bash
   cp /etc/wireguard/wg0.conf.example /etc/wireguard/wg0.conf
   chmod 0600 /etc/wireguard/wg0.conf
   # Edit /etc/wireguard/wg0.conf and replace all <...> placeholders
   ```

2. **(Optional) If enabling the standalone `killswitch.nft` file**:
   ```bash
   cp /opt/telegram-group-bot/deploy/wireguard/killswitch.nft.example /etc/wireguard/killswitch.nft
   chmod 0600 /etc/wireguard/killswitch.nft
   nft -c -f /etc/wireguard/killswitch.nft
   ```

3. **Test interactively with a safety rollback timer** (keeps SSH safe if endpoint/routing is misconfigured):
   ```bash
   # In root shell on 192.168.1.164:
   systemd-run --on-active=60s /usr/bin/wg-quick down wg0
   wg-quick up wg0
   ```
   Wait — verify connectivity in the active session:
   ```bash
   wg show
   ip -4 rule
   ip route get 192.168.1.1
   ip route get 1.1.1.1
   curl -fsS https://ifconfig.me && echo
   # If kill-switch enabled:
   nft list table inet wg_killswitch
   ```
   If everything looks good before the 60s timer fires, bring it down cleanly (`wg-quick down wg0`) or cancel the rollback timer (`systemctl stop run-*.timer` / `wg-quick down wg0` after confirming).

4. **Check both bots remain healthy**:
   ```bash
   systemctl is-active telegram-userbot.service telegram-mtproto-userbot.service
   ```

5. **Enable persistence across reboots (only after manual verification succeeds)**:
   ```bash
   systemctl enable --now wg-quick@wg0
   ```

6. **Rollback / Disable at any time**:
   ```bash
   systemctl disable --now wg-quick@wg0
   nft delete table inet wg_killswitch 2>/dev/null || true
   ```
