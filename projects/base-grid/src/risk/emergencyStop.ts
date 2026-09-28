import { existsSync } from "node:fs";
import { join } from "node:path";

export class EmergencyStop {
  private active = false;
  private reason?: string;

  isActive(): boolean {
    return this.active;
  }

  getReason(): string | undefined {
    return this.reason;
  }

  checkFileTrigger(projectRoot: string): boolean {
    const stopFilePath = join(projectRoot, "stop");
    if (existsSync(stopFilePath)) {
      this.activate("Stop file detected: " + stopFilePath);
      return true;
    }
    return false;
  }

  checkEnvTrigger(): boolean {
    if (process.env.EMERGENCY_STOP === "true" || process.env.EMERGENCY_STOP === "1") {
      this.activate("EMERGENCY_STOP env variable set");
      return true;
    }
    return false;
  }

  checkBalanceTrigger(ethBalance: bigint, minReserve: bigint): boolean {
    if (ethBalance < minReserve) {
      this.activate(
        `ETH balance ${ethBalance.toString()} below minimum reserve ${minReserve.toString()}`,
      );
      return true;
    }
    return false;
  }

  activate(reason: string): void {
    this.active = true;
    this.reason = reason;
  }

  deactivate(): void {
    this.active = false;
    this.reason = undefined;
  }

  registerSignalHandlers(onStop: () => Promise<void>): void {
    const handler = async (signal: string) => {
      this.activate(`Received ${signal}`);
      await onStop();
      process.exit(0);
    };

    process.on("SIGINT", () => handler("SIGINT"));
    process.on("SIGTERM", () => handler("SIGTERM"));
  }
}
